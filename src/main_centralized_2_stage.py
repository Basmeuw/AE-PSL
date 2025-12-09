from argparse import Namespace

import torch
from torch.utils.data import Subset

import available_datasets as datasets
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
from utils.cuda_utils import get_free_cuda_device_name
from utils.model_saving_utils import save_centralized_model, \
    save_experiment_results, save_ae_experiment_results
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
    device = torch.device(get_free_cuda_device_name(global_args) if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    full_dataset = datasets.load_data(name=global_args['dataset'], num_partitions=1, split='iid',
                                      seed=global_args['random_seed'], global_args=global_args)
    train_data = full_dataset.load_partition(partition_id=0)

    if global_args['small_test_run']: train_data = datasets.Subset(full_dataset, range(0, len(full_dataset) // 20))
    train_dataloader = datasets.DataLoader(train_data, batch_size=global_args['batch_size'], shuffle=True, pin_memory=True,
                                           num_workers=global_args['num_workers'],
                                           collate_fn=full_dataset.get_collate_fn())

    test_ds = full_dataset.load_test_set()
    if global_args['small_test_run']: test_ds = datasets.Subset(test_ds, range(0, len(test_ds) // 20))
    test_dataloader = datasets.DataLoader(test_ds, batch_size=global_args['batch_size'], shuffle=False, pin_memory=True,
                                          num_workers=global_args['num_workers'], collate_fn=full_dataset.get_collate_fn())

    # First load the base model, which is used to retrieve intermediate activations for the auto-encoder training
    base_model = get_base_model(global_args, device)
    base_model.to(device)
    # ============== AE Pre-training ==============
    # This signature is used to identify the AE model. It will be used to find an existing AE model or to save a new one.
    ae_signature = {
        'type': global_args['ae_type'],
        'dataset': global_args['ae_pretrain_dataset'],
        'dataset_proportion': global_args['ae_pretrain_dataset_fraction'],
        'model': global_args['model'],
        'split_layer': global_args['split_layer'],
        'input_dim': base_model.get_hidden_dim(),
        'latent_dim': global_args['ae_latent_dim'],
    }

    auto_encoder_model = initialize_AE(global_args, base_model.get_hidden_dim())
    auto_encoder_model.to(device)

    auto_encoder_trainer = CentralizedAETrainer()

    if type(auto_encoder_model) is IdentityAE:
        print("AE type is IdentityAE, skipping AE pretraining.")
    else:
        if global_args['ae_use_existing']:
            # Try to load an existing AE model
            auto_encoder_model = load_auto_encoder_model(global_args, auto_encoder_model, ae_signature, device)
        else:
            # Train a new AE model
            optimizer, scheduler = get_ae_pretrain_optimizer_and_scheduler(auto_encoder_model, global_args)
            loss_fn = torch.nn.MSELoss()  # make this configurable later

            # A utility object that can be used to streamline archiving experiment results.
            ae_experiment_results = ExperimentResultsAE()
            prepare_ae_dir(ae_signature)

            # # # # # # # # # # # # # # # # # Training # # # # # # # # # # # # # # # # #
            for epoch_nr in range(global_args['ae_pretrain_epochs']):
                print(f'Starting AE epoch {epoch_nr + 1}')
                warnings.warn("Warning: using the same train dataloader for AE pretraining as for the main training.")
                # TODO do partial dataset for AE pretraining based on param
                train_results_as_string = auto_encoder_trainer.train_epoch(
                    experiment_results=ae_experiment_results,
                    auto_encoder=auto_encoder_model,
                    base_model=base_model,
                    split_layer=global_args['split_layer'],
                    device=device,
                    dataloader=train_dataloader,
                    epoch_nr=epoch_nr + 1,
                    optimizer=optimizer,
                    loss_fn=loss_fn
                )
                print(train_results_as_string)

                scheduler.step()

                with torch.no_grad():
                    print('Starting AE evaluation')
                    # NOTE: we are now using a different dataset to extract activations, unlike in feasibility experiment
                    test_results_as_string = auto_encoder_trainer.train_epoch(
                        experiment_results=ae_experiment_results,
                        auto_encoder=auto_encoder_model,
                        base_model=base_model,
                        split_layer=global_args['split_layer'],
                        device=device,
                        dataloader=test_dataloader,
                        epoch_nr=epoch_nr + 1,
                        optimizer=None,
                        loss_fn=loss_fn
                    )

                    print(test_results_as_string)

                save_ae_experiment_results(ae_experiment_results, filename_from_signature(ae_signature))

            print(f'Finished training AE - Saving final model')

            if global_args['ae_save_final_weights']:
                save_ae_with_signature(auto_encoder_model, ae_signature)

    if global_args['ae_pretrain_only']:
        print("Skipping stage 2 as per user request.")
        return

    # Using the base model and the AE, load the full centralized model and the trainer
    full_model, trainer = get_centralized_model_and_trainer(global_args, device, base_model=base_model,
                                                            auto_encoder=auto_encoder_model)
    full_model = full_model.switch_to_device(device)

    # enable profiling
    # trainer = ProfiledTrainer()

    print(
        f'trainable params centralized model: {sum(p.numel() for p in full_model.parameters() if p.requires_grad)} | all: {sum(p.numel() for p in full_model.parameters())}')

    optimizer, scheduler = get_optimizer_and_scheduler(full_model, global_args)

    # A utility object that can be used to streamline archiving experiment results.
    experiment_results = ExperimentResults()
    # Make sure to save any specific hyperparameters used for this experiment
    experiment_results.params = search_space_args

    # # # # # # # # # # # # # # # # # Stage 2 Training # # # # # # # # # # # # # # # # #
    for epoch_nr in range(global_args['nr_of_epochs']):
        print(f'Starting epoch {epoch_nr + 1}')

        train_results_as_string = handle_train_epoch(trainer, experiment_results, full_model, device, train_dataloader,
                                                     epoch_nr + 1, optimizer)
        print(train_results_as_string)

        scheduler.step()

        if global_args['save_model_after_each_epoch']:
            save_centralized_model(full_model, global_args['save_file_name'])

        with torch.no_grad():
            print('Starting evaluation')
            test_results_as_string = handle_test_epoch(trainer, experiment_results, full_model, device, test_dataloader,
                                                       epoch_nr + 1)

            print(test_results_as_string)

        save_experiment_results(experiment_results, global_args['save_file_name'])

    print(f'Finished training - Saving final model')

    if global_args['save_final_model']:
        save_centralized_model(full_model, global_args['save_file_name'])


if __name__ == '__main__':
    # print(global_args)
    run_2_stage(setup_arguments(), None)



