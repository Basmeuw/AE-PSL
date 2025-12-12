import copy
import threading
import warnings
from argparse import Namespace

import torch

import available_datasets as datasets
import models
from ae_trainers.ae_trainer import pretrain_auto_encoder, get_auto_encoder
from main_centralized_2_stage import run_2_stage
from models import IdentityAE
from models import get_base_model
from trainers.implementations.experiment_results import ExperimentResults
from trainers.implementations.experiment_trainer import ExperimentTrainer
from utils.ae_registry_utils import load_auto_encoder_model
from utils.argument_utils import build_base_argument_parser, \
    expand_argument_parser_with_distributed_learning_parameters, validate_base_argument_constraints, set_env_variables, \
    namespace_to_dict
from utils.config_utils import set_random_seed
from utils.cuda_utils import get_free_cuda_device_name, get_device
from utils.dataloader_utils import get_distributed_dataloaders_from_datasets
from utils.fl_utils import fed_avg, get_client_weight_multipliers__nr_of_elements
from utils.model_saving_utils import save_split_model, save_experiment_results
from utils.mpsl_utils import compute_mini_batch_size
from utils.scheduler_utils import get_optimizer_and_scheduler


def init_data_iterator(client_id, client_data_iterators, client_dataloaders, barrier):
    client_data_iterators[client_id] = iter(client_dataloaders[client_id])
    barrier.wait()


def handle_train_epoch(
        trainer: ExperimentTrainer,
        experiment_results,
        device,
        server_model,
        server_optimizer,
        server_scheduler,
        client_dataloaders,
        client_models,
        client_model_requires_any_grad,
        max_nr_of_batches_in_epoch,
        client_optimizers,
        client_schedulers,
        epoch_nr,
        global_args):
    client_data_iterators = dict()

    # Note: Data iterators are initialized in separate threads to speed up loading time on slow disks with a large number of clients.
    print('Initializing train data iterators')
    barrier = threading.Barrier(global_args['nr_of_clients'] + 1)

    for client_id in range(global_args['nr_of_clients']):
        threading.Thread(target=init_data_iterator, args=(client_id, client_data_iterators, client_dataloaders, barrier)).start()

    barrier.wait()

    return trainer.train_epoch(
        device=device,
        experiment_results=experiment_results,
        server_model=server_model,
        server_optimizer=server_optimizer,
        server_scheduler=server_scheduler,
        client_data_iterators=client_data_iterators,
        client_models=client_models,
        client_model_requires_any_grad=client_model_requires_any_grad,
        client_optimizers=client_optimizers,
        client_schedulers=client_schedulers,
        max_nr_of_batches_in_epoch=max_nr_of_batches_in_epoch,
        epoch_nr=epoch_nr,
        global_args=global_args
    )


def handle_test_epoch(trainer, experiment_results, device, client_model, server_model, dataloader, epoch_nr):
    return trainer.test_epoch(
        device=device,
        experiment_results=experiment_results,
        server_model=server_model,
        client_model=client_model,
        dataloader=dataloader,
        epoch_nr=epoch_nr
    )


def setup_arguments() -> dict:
    parser = build_base_argument_parser()
    parser = expand_argument_parser_with_distributed_learning_parameters(parser)

    parser.add_argument('--test_num_workers', type=int, default=5,
                        help='num_workers provided to the test Dataloader. For Split Learning, we differentiate between num_workers for the train Dataloader and test_num_workers for the test Dataloader.')

    args: Namespace = parser.parse_args()

    validate_base_argument_constraints(args)

    set_env_variables(args)

    return namespace_to_dict(args)


def run_2_stage_mpsl(global_args: dict, search_space_args: dict):
    set_random_seed(global_args['random_seed'])

    # # # # # # # # # # # # # # # # # Setup # # # # # # # # # # # # # # # # #
    device = get_device(global_args)
    mini_batch_size = compute_mini_batch_size(global_args['batch_size'], global_args['nr_of_clients'])
    total_batch_size = global_args['nr_of_clients'] * mini_batch_size

    print(f'Device: {device}')
    print(
        f'Available datasets: {datasets.available_datasets()}, chosen: {global_args["dataset"]} with batch_size: {global_args["batch_size"]} and mini-batch size: {mini_batch_size}. Hence real total batch_size is {total_batch_size}')

    if total_batch_size > global_args['batch_size']:
        raise Exception(f'total_batch_size > chosen batch_size ({total_batch_size} > {global_args["batch_size"]})')

    validation_mode = global_args['val_mode']
    validation_split = global_args['val_split']
    if validation_mode == 'none': validation_split = 0.0

    full_dataset = datasets.load_data(name=global_args['dataset'], num_partitions=global_args['nr_of_clients'],
                                      split=global_args['dataset_split_type'], seed=global_args['random_seed'],
                                      global_args=global_args, val_split=validation_split)

    val_dataset = full_dataset.load_validation_set()
    test_dataset = full_dataset.load_test_set()

    base_model = get_base_model(global_args, device=device)

    is_train_ae_on_downstream_data = global_args['dataset'] == global_args['ae_pretrain_dataset']

    # We only reuse the full dataset if AE pretraining uses the same dataset
    auto_encoder_model = get_auto_encoder(global_args, base_model, device,
                                          full_dataset if is_train_ae_on_downstream_data else None)

    if global_args['ae_pretrain_only']:
        print("Skipping fine-tuning as ae_pretrain_only is set to true")
        return



    client_dataloaders, val_dataloader, test_dataloader = get_distributed_dataloaders_from_datasets(
        train_dataset=full_dataset,
        validation_dataset=val_dataset,
        test_dataset=test_dataset,
        validation_mode=validation_mode,
        mini_batch_size=mini_batch_size,
        total_batch_size=total_batch_size,
        num_workers=global_args['num_workers'],
        dataloader_class=datasets.DataLoader,
        nr_of_clients=global_args['nr_of_clients'],
        small_test_run=global_args['small_test_run'],
        collate_fn=full_dataset.get_collate_fn()
    )

    (client_model, server_model, client_model_requires_any_grad), trainer = models.get_split_model_pair_and_trainer(
        global_args, device, base_model, auto_encoder_model)
    server_model = server_model.switch_to_device(device)


    client_models = dict()

    client_optimizers = dict()
    client_schedulers = dict()

    max_nr_of_batches_in_epoch = 0

    print(
        f'Number of trainable params server model: {sum(p.numel() for p in server_model.parameters() if p.requires_grad)} | Total number of parameters: {sum(p.numel() for p in server_model.parameters())}')
    print(
        f'Number of trainable params client model: {sum(p.numel() for p in client_model.parameters() if p.requires_grad)} | Total number of parameters: {sum(p.numel() for p in client_model.parameters())}')

    for client_id in range(global_args['nr_of_clients']):
        _client_model = client_model if client_id == 0 else copy.deepcopy(client_model)
        _client_model = _client_model.switch_to_device(device)
        client_optimizer, client_scheduler = get_optimizer_and_scheduler(_client_model, global_args)
        client_optimizers[client_id] = client_optimizer
        client_models[client_id] = _client_model

        if client_model_requires_any_grad:
            client_schedulers[client_id] = client_scheduler

        max_nr_of_batches_in_epoch = max(max_nr_of_batches_in_epoch, len(client_dataloaders[client_id]))

    server_optimizer, server_scheduler = get_optimizer_and_scheduler(server_model, global_args)

    # A utility object that can be used to streamline archiving experiment results.
    experiment_results = ExperimentResults('none')
    experiment_results.params = search_space_args

    # # # # # # # # # # # # # # # # # Training # # # # # # # # # # # # # # # # #
    for epoch_nr in range(global_args['nr_of_epochs']):
        print(f'Starting epoch {epoch_nr + 1}')

        train_loss, acc, nr_of_elements_per_client_dict = handle_train_epoch(
            trainer,
            experiment_results,
            device,
            server_model,
            server_optimizer,
            server_scheduler,
            client_dataloaders,
            client_models,
            client_model_requires_any_grad,
            max_nr_of_batches_in_epoch,
            client_optimizers,
            client_schedulers,
            epoch_nr + 1,
            global_args
        )
        print(f'Finished epoch {epoch_nr} with train loss {train_loss} and accuracy {acc}')

        server_scheduler.step()

        aggregated_client_model = fed_avg(client_models,
                                          get_client_weight_multipliers__nr_of_elements(nr_of_elements_per_client_dict))

        if global_args['save_model_after_each_epoch']:
            save_split_model(aggregated_client_model, server_model, global_args['save_file_name'])

        test_loss, acc = handle_test_epoch(
            trainer,
            experiment_results,
            device,
            aggregated_client_model,
            server_model,
            test_dataloader,
            epoch_nr + 1
        )
        print(f'test loss: {test_loss}, test accuracy: {acc}')

        save_experiment_results(experiment_results, global_args['save_file_name'])

    print(f'Finished training - saving final model')

    if global_args['save_final_model']:
        save_split_model(aggregated_client_model, server_model, global_args['save_file_name'])
#
# def finetune_distributed(global_args,
#             trainer,
#             experiment_results,
#             device,
#             server_model,
#             client_dataloaders,
#             client_models,
#             client_model_requires_any_grad,
#             max_nr_of_batches_in_epoch,
#             client_optimizers,
#             client_schedulers,
#              search_space_args,
#                              validation_mode, early_stopping_patience=3):
#     """
#         Handles the case for early stopping / saving
#
#         Args:
#             validation_mode (str): 'none', 'early_saving', or 'early_stopping'.
#         """
#
#     server_optimizer, server_scheduler = get_optimizer_and_scheduler(server_model, global_args)
#
#     # A utility object that can be used to streamline archiving experiment results.
#     experiment_results = ExperimentResults(validation_mode=validation_mode)
#     # Make sure to save any specific hyperparameters used for this experiment
#     experiment_results.params = search_space_args  # Ensure search_space_args is in scope or passed in
#
#     # Validation State Variables
#     best_val_loss = float('inf')
#     patience_counter = 0
#
#     # # # # # # # # # # # # # # # # # Stage 2 Training # # # # # # # # # # # # # # # # #
#     for epoch_nr in range(global_args['nr_of_epochs']):
#         print(f'Starting epoch {epoch_nr + 1}')
#
#         train_loss, acc, nr_of_elements_per_client_dict = handle_train_epoch(
#             trainer,
#             experiment_results,
#             device,
#             server_model,
#             server_optimizer,
#             server_scheduler,
#             client_dataloaders,
#             client_models,
#             client_model_requires_any_grad,
#             max_nr_of_batches_in_epoch,
#             client_optimizers,
#             client_schedulers,
#             epoch_nr + 1,
#             global_args
#         )
#         print(f'Finished epoch {epoch_nr} with train loss {train_loss} and accuracy {acc}')
#         scheduler.step()
#
#         # Regular checkpointing (unrelated to early stopping)
#         if global_args['save_model_after_each_epoch']:
#             save_centralized_model(full_model, global_args['save_file_name'])
#
#         with torch.no_grad():
#
#             # --- Option 2 & 3: Use Validation Set ---
#             if val_dataloader is not None and validation_mode in ['early_saving', 'early_stopping']:
#                 print('Starting Validation')
#
#                 # NOTE: handle_test_epoch MUST return (loss, string) for this to work
#                 val_loss, val_msg = handle_test_epoch(trainer=trainer,
#                                                       experiment_results=experiment_results,
#                                                       model=full_model,
#                                                       device=device,
#                                                       dataloader=val_dataloader,
#                                                       epoch_nr=global_args['nr_of_epochs'] + 1)
#                 print(f"Validation: {val_msg}")
#
#                 # Check for improvement
#                 if val_loss < best_val_loss:
#                     best_val_loss = val_loss
#                     patience_counter = 0
#                     print(f"Validation loss improved to {best_val_loss:.6f}. Saving model...")
#                     save_centralized_model(full_model, global_args['save_file_name'])
#                 else:
#                     patience_counter += 1
#                     print(f"Validation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")
#
#                 # Early Stopping Logic
#                 if validation_mode == 'early_stopping' and patience_counter >= early_stopping_patience:
#                     print(f"Early stopping triggered after {epoch_nr + 1} epochs.")
#                     break
#
#             # --- Option 1: No Validation (Legacy Behavior) ---
#             elif validation_mode == 'none':
#                 print('Starting Test Set Evaluation (Logging only)')
#                 # Legacy behavior: Run test set every epoch
#                 # trainer, experiment_results, model, device, dataloader, epoch_nr, optimizer
#                 test_loss, acc = handle_test_epoch(
#                     trainer,
#                     experiment_results,
#                     device,
#                     aggregated_client_model,
#                     server_model,
#                     test_dataloader,
#                     epoch_nr + 1
#                 )
#                 print(test_results[1])
#
#         save_experiment_results(experiment_results, global_args['save_file_name'])
#
#     print(f'Finished training.')
#
#     # --- Final Test Set Run ---
#     # If we used early saving/stopping, we must load the best model back in
#     if validation_mode in ['early_saving', 'early_stopping']:
#         print(f"Loading best model for final testing...")
#         try:
#             load_centralized_model(full_model, device, global_args['save_file_name'])
#         except FileNotFoundError:
#             print("Warning: Best model file not found. Using current weights.")
#
#     print('Starting Final Test Set Evaluation')
#     with torch.no_grad():
#         # trainer, experiment_results, model, device, dataloader, epoch_nr, optimizer
#         final_test_metric, final_test_msg = handle_test_epoch(trainer=trainer,
#                                                               experiment_results=experiment_results,
#                                                               model=full_model,
#                                                               device=device,
#                                                               dataloader=test_dataloader,
#                                                               epoch_nr=global_args['nr_of_epochs'] + 1)
#
#         experiment_results.final_test_metric = final_test_metric
#         print(final_test_msg)
#
#     # Save final model as per global args (usually for the artifact storage)
#     if global_args['save_final_model']:
#         save_centralized_model(full_model, global_args['save_file_name'])
#
#     return full_model


if __name__ == '__main__':
    global_args = setup_arguments()
    run_2_stage_mpsl(global_args, None)


