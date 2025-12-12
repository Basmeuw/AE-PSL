from argparse import Namespace

import torch
from torch.utils.data import Subset

import available_datasets as datasets
from ae_trainers.ae_trainer import pretrain_auto_encoder, get_auto_encoder
from ae_trainers.implementations.ae_experiment_results import ExperimentResultsAE
from ae_trainers.implementations.classification.centralized_ae_trainer import CentralizedAETrainer
from models import get_centralized_model_and_trainer, IdentityAE, get_base_model
from models.auto_encoder import initialize_AE

from trainers.implementations.classification.profiled_centralized_trainer import ProfiledTrainer
from trainers.implementations.experiment_results import ExperimentResults
from utils.ae_registry_utils import load_auto_encoder_model, load_auto_encoder_model, filename_from_signature, \
    save_ae_with_signature, prepare_ae_dir
from utils.argument_utils import build_base_argument_parser, validate_base_argument_constraints, \
    expand_argument_parser_with_adapter_approach_parameters, set_env_variables, \
    expand_argument_parser_with_ae_pretraining_parameters, namespace_to_dict
from utils.config_utils import set_random_seed
from utils.cuda_utils import get_free_cuda_device_name, get_device
from utils.dataloader_utils import get_dataloaders_from_datasets
from utils.model_saving_utils import save_centralized_model, \
    save_experiment_results, save_ae_experiment_results, load_centralized_model
from utils.scheduler_utils import get_optimizer_and_scheduler, get_ae_pretrain_optimizer_and_scheduler
import warnings


def handle_train_epoch(trainer, experiment_results, model, device, dataloader, epoch_nr, optimizer):
    return trainer.train_epoch(
        experiment_results=experiment_results,
        model=model,
        device=device,
        dataloader=dataloader,
        epoch_nr=epoch_nr,
        optimizer=optimizer
    )


def handle_test_epoch(trainer, experiment_results, model, device, dataloader, epoch_nr):
    return trainer.test_epoch(
        experiment_results=experiment_results,
        model=model,
        device=device,
        dataloader=dataloader,
        epoch_nr=epoch_nr
    )


def setup_arguments() -> dict:
    parser = build_base_argument_parser()
    parser = expand_argument_parser_with_ae_pretraining_parameters(parser)

    args: Namespace = parser.parse_args()

    validate_base_argument_constraints(args)
    set_env_variables(args)

    return namespace_to_dict(args)


def run_2_stage(global_args: dict, search_space_args: dict):
    set_random_seed(global_args['random_seed'])

    # # # # # # # # # # # # # # # # # Setup # # # # # # # # # # # # # # # # #
    device = get_device(global_args)
    print(f'Device: {device}')

    full_dataset = datasets.load_data(name=global_args['dataset'], num_partitions=1, split='iid',
                                      seed=global_args['random_seed'], global_args=global_args)

    # First load the base model, which is used to retrieve intermediate activations for the auto-encoder training
    base_model = get_base_model(global_args, device)
    base_model.to(device)

    is_train_ae_on_downstream_data = global_args['dataset'] == global_args['ae_pretrain_dataset']

    # We only reuse the full dataset if AE pretraining uses the same dataset
    auto_encoder_model = get_auto_encoder(global_args, base_model, device, full_dataset if is_train_ae_on_downstream_data else None)

    if global_args['ae_pretrain_only']:
        print("Skipping fine-tuning as ae_pretrain_only is set to true")
        return

    # Load partitions for fine-tuning
    train_data = full_dataset.load_partition(partition_id=0)
    test_data = full_dataset.load_test_set()


    validation_mode = global_args['val_mode']
    validation_split = global_args['val_split']
    if validation_split == 0: validation_mode = 'none'

    # If user wants to use validation set (for early stopping or saving), it will be provided as a subset of the train set
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders_from_datasets(train_data, test_data,
                                                                                      validation_mode, validation_split,
                                                                                      global_args['batch_size'],
                                                                                      global_args['num_workers'],
                                                                                      datasets.DataLoader,
                                                                                      global_args['small_test_run'],
                                                                                      collate_fn=full_dataset.get_collate_fn())

    # Using the base model and the AE, load the full centralized model and the trainer
    full_model, trainer = get_centralized_model_and_trainer(global_args, device, base_model=base_model,
                                                            auto_encoder=auto_encoder_model)
    full_model = full_model.switch_to_device(device)

    finetune_centralized(global_args, device, full_model, trainer, train_dataloader, val_dataloader,
                          test_dataloader, search_space_args, validation_mode)



def finetune_centralized(global_args, device, full_model, trainer,
                             train_dataloader, val_dataloader, test_dataloader, search_space_args,
                             validation_mode, early_stopping_patience=3):
    """
    Handles the case for early stopping / saving

    Args:
        validation_mode (str): 'none', 'early_saving', or 'early_stopping'.
    """
    print(
        f'trainable params centralized model: {sum(p.numel() for p in full_model.parameters() if p.requires_grad)} | all: {sum(p.numel() for p in full_model.parameters())}')

    optimizer, scheduler = get_optimizer_and_scheduler(full_model, global_args)

    # A utility object that can be used to streamline archiving experiment results.
    experiment_results = ExperimentResults(validation_mode=validation_mode)
    # Make sure to save any specific hyperparameters used for this experiment
    experiment_results.params = search_space_args  # Ensure search_space_args is in scope or passed in

    # Validation State Variables
    best_val_loss = float('inf')
    patience_counter = 0

    # # # # # # # # # # # # # # # # # Stage 2 Training # # # # # # # # # # # # # # # # #
    for epoch_nr in range(global_args['nr_of_epochs']):
        print(f'Starting epoch {epoch_nr + 1}')

        train_loss, train_msg = handle_train_epoch(trainer=trainer,
                                               experiment_results=experiment_results,
                                               model=full_model,
                                               device=device,
                                               dataloader=train_dataloader,
                                               epoch_nr=global_args['nr_of_epochs'] + 1,
                                                optimizer=optimizer)
        print(train_msg)

        scheduler.step()

        # Regular checkpointing (unrelated to early stopping)
        if global_args['save_model_after_each_epoch']:
            save_centralized_model(full_model, global_args['save_file_name'])

        with torch.no_grad():

            # --- Option 2 & 3: Use Validation Set ---
            if val_dataloader is not None and validation_mode in ['early_saving', 'early_stopping']:
                print('Starting Validation')

                # NOTE: handle_test_epoch MUST return (loss, string) for this to work
                val_loss, val_msg = handle_test_epoch(trainer=trainer,
                                               experiment_results=experiment_results,
                                               model=full_model,
                                               device=device,
                                               dataloader=val_dataloader,
                                               epoch_nr=global_args['nr_of_epochs'] + 1)
                print(f"Validation: {val_msg}")

                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    print(f"Validation loss improved to {best_val_loss:.6f}. Saving model...")
                    save_centralized_model(full_model, global_args['save_file_name'])
                else:
                    patience_counter += 1
                    print(f"Validation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")

                # Early Stopping Logic
                if validation_mode == 'early_stopping' and patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch_nr + 1} epochs.")
                    break

            # --- Option 1: No Validation (Legacy Behavior) ---
            elif validation_mode == 'none':
                print('Starting Test Set Evaluation (Logging only)')
                # Legacy behavior: Run test set every epoch
                #trainer, experiment_results, model, device, dataloader, epoch_nr, optimizer
                test_results = handle_test_epoch(trainer=trainer,
                                               experiment_results=experiment_results,
                                               model=full_model,
                                               device=device,
                                               dataloader=test_dataloader,
                                               epoch_nr=global_args['nr_of_epochs'] + 1)
                print(test_results[1])

        save_experiment_results(experiment_results, global_args['save_file_name'])

    print(f'Finished training.')

    # --- Final Test Set Run ---
    # If we used early saving/stopping, we must load the best model back in
    if validation_mode in ['early_saving', 'early_stopping']:
        print(f"Loading best model for final testing...")
        try:
            load_centralized_model(full_model, device, global_args['save_file_name'])
        except FileNotFoundError:
            print("Warning: Best model file not found. Using current weights.")

    print('Starting Final Test Set Evaluation')
    with torch.no_grad():
        #trainer, experiment_results, model, device, dataloader, epoch_nr, optimizer
        final_test_metric, final_test_msg = handle_test_epoch(trainer=trainer,
                                               experiment_results=experiment_results,
                                               model=full_model,
                                               device=device,
                                               dataloader=test_dataloader,
                                               epoch_nr=global_args['nr_of_epochs'] + 1)

        experiment_results.final_test_metric = final_test_metric
        print(final_test_msg)

    # Save final model as per global args (usually for the artifact storage)
    if global_args['save_final_model']:
        save_centralized_model(full_model, global_args['save_file_name'])

    return full_model

if __name__ == '__main__':
    # print(global_args)
    run_2_stage(setup_arguments(), None)



