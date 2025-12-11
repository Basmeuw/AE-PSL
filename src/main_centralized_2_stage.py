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
    device = get_device(global_args)
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

    is_train_ae_on_downstream_data = global_args['dataset'] == global_args['ae_pretrain_dataset']

    # We only reuse the full dataset if AE pretraining uses the same dataset
    auto_encoder_model = get_auto_encoder(global_args, base_model, device, full_dataset if is_train_ae_on_downstream_data else None)

    if global_args['ae_pretrain_only']:
        print("Skipping fine-tuning as ae_pretrain_only is set to true")
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
            # TODO use validation set instead of test set for intermediate evals
            # TODO use early stopping?

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



