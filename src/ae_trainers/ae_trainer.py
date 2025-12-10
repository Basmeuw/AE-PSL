import warnings

import torch

from ae_trainers.implementations.ae_experiment_results import ExperimentResultsAE
from ae_trainers.implementations.classification.centralized_ae_trainer import CentralizedAETrainer
from models.auto_encoder import initialize_AE, IdentityAE
from utils.ae_registry_utils import load_auto_encoder_model, prepare_ae_dir, filename_from_signature, \
    save_ae_with_signature
from utils.model_saving_utils import save_ae_experiment_results
from utils.scheduler_utils import get_ae_pretrain_optimizer_and_scheduler


def pretrain_auto_encoder(global_args, base_model, train_dataloader, test_dataloader, device):

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

    return auto_encoder_model