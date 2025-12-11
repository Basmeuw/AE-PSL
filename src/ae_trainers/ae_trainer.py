import warnings

import torch
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import available_datasets
from ae_trainers.implementations.ae_experiment_results import ExperimentResultsAE
from ae_trainers.implementations.classification.centralized_ae_trainer import CentralizedAETrainer
from available_datasets import DistributedDataset
from models import InputModality
from models.auto_encoder import initialize_AE, IdentityAE
from utils.ae_registry_utils import load_auto_encoder_model, prepare_ae_dir, filename_from_signature, \
    save_ae_with_signature
from utils.model_saving_utils import save_ae_experiment_results
from utils.scheduler_utils import get_ae_pretrain_optimizer_and_scheduler


activations = {}

# Makes the choice to pretrain or load an AE model based on global_args
def get_auto_encoder(global_args, base_model, device, full_dataset : DistributedDataset = None):

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

    # Initialize an untrained AE model
    auto_encoder_model = initialize_AE(global_args, base_model.get_hidden_dim())
    auto_encoder_model.to(device)

    if type(auto_encoder_model) is IdentityAE:
        print("AE type is IdentityAE, skipping AE pretraining/loading.")
        return auto_encoder_model

    if global_args['ae_use_existing']:
        # Load an existing AE model based on required ae signature
        return load_auto_encoder_model(global_args, auto_encoder_model, ae_signature, device)


    # If the AE pretrain dataset differs, then we are training on downstream client data
    is_train_ae_on_downstream_data = global_args['dataset'] == global_args['ae_pretrain_dataset']

    if is_train_ae_on_downstream_data:
        print(f"Note: AE pre-training on downstream client dataset: {global_args['dataset']}")
    else:
        print(f"Note: AE pre-training on separate pre-train dataset: {global_args['ae_pretrain_dataset']}")

    # If we are using MPSL, we might have to pretrain the AEs on the client-side
    # This requires extra logic, as we need multiple copies of the AEs, and something needs to be done to aggregate the AEs,
    # or keep server-side per-client AEs.
    if global_args['train_method'] == 'mpsl' and is_train_ae_on_downstream_data:
        raise NotImplementedError("AE pretraining on client-side / downstream data for MPSL not yet implemented.")
    else:
        # else, we simply do centralized pretraining on a specific dataset
        # Load data and pretrain on the specified dataset

        if global_args['train_method'] == 'mpsl':
            print(
                "Note: currently not reusing the specified full_dataset, as it might be split into partitions because we use MPSL.")
            full_dataset = available_datasets.load_data(name=global_args['dataset'], num_partitions=global_args['nr_of_clients'],
                                              split=global_args['dataset_split_type'], seed=global_args['random_seed'],
                                              global_args=global_args)
        else:
            # can't reuse the specified full_dataset, since it is None, and is different to the ae pretrain dataset
            if not is_train_ae_on_downstream_data:
                full_dataset = available_datasets.load_data(name=global_args['dataset'], num_partitions=global_args['nr_of_clients'],
                                                  split=global_args['dataset_split_type'], seed=global_args['random_seed'],
                                                global_args=global_args)
        train_dataset = full_dataset.load_partition(partition_id=0)
        test_dataset = full_dataset.load_test_set()

        if global_args['small_test_run']:
            train_dataset = available_datasets.Subset(train_dataset, range(0, len(full_dataset) // 20))
            test_dataset = available_datasets.Subset(test_dataset, range(0, len(full_dataset) // 20))
        else:
            # Take the correct proportion of the dataset, as we want to experiment with how much data is needed to pretrain the AE
            train_dataset = available_datasets.Subset(train_dataset, range(0, int(len(full_dataset) * global_args[
                'ae_pretrain_dataset_fraction'])))
            test_dataset = available_datasets.Subset(test_dataset, range(0, int(len(full_dataset) * global_args[
                'ae_pretrain_dataset_fraction'])))

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=global_args['ae_pretrain_batch_size'], shuffle=True,
                                      pin_memory=True,
                                      num_workers=global_args['num_workers'],
                                      collate_fn=full_dataset.get_collate_fn())

        test_dataloader = DataLoader(test_dataset,
                                     batch_size=global_args['ae_pretrain_batch_size'], shuffle=False,
                                     pin_memory=True,
                                     num_workers=global_args['num_workers'],
                                     collate_fn=full_dataset.get_collate_fn())

        # Now we load retrieve the activations
        # TODO we can cache this to avoid recomputing if needed again, but this is easier for now
        train_acts_dataset = extract_activations(
            base_model, global_args['split_layer'], train_dataloader, device
        )
        test_acts_dataset = extract_activations(
            base_model, global_args['split_layer'], test_dataloader, device
        )

        return pretrain_auto_encoder(global_args, base_model, auto_encoder_model, train_acts_dataset, test_acts_dataset, ae_signature, device)



def pretrain_auto_encoder(global_args, base_model, auto_encoder_model, train_acts_dataset, test_acts_dataset, ae_signature, device):

    # Don't need to use the custom distributed dataset class here, as we are using activations only.
    train_acts_dataloader = DataLoader(train_acts_dataset, batch_size=global_args['ae_pretrain_batch_size'], shuffle=True, pin_memory=True, num_workers=global_args['num_workers'])
    test_acts_dataloader = DataLoader(test_acts_dataset, batch_size=global_args['ae_pretrain_batch_size'], shuffle=True, pin_memory=True, num_workers=global_args['num_workers'])

    auto_encoder_trainer = CentralizedAETrainer()

    optimizer, scheduler = get_ae_pretrain_optimizer_and_scheduler(auto_encoder_model, global_args)
    loss_fn = torch.nn.MSELoss()  # make this configurable later

    # A utility object that can be used to streamline archiving experiment results.
    ae_experiment_results = ExperimentResultsAE()
    prepare_ae_dir(ae_signature)

    # # # # # # # # # # # # # # # # # Training # # # # # # # # # # # # # # # # #
    for epoch_nr in range(global_args['ae_pretrain_epochs']):
        print(f'Starting AE epoch {epoch_nr + 1}')

        train_results_as_string = auto_encoder_trainer.train_epoch(
            experiment_results=ae_experiment_results,
            auto_encoder=auto_encoder_model,
            base_model=base_model,
            split_layer=global_args['split_layer'],
            device=device,
            dataloader=train_acts_dataloader,
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
                dataloader=test_acts_dataloader,
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

def extract_activations(base_model, split_layer, dataloader, device):
    """
    Runs the base_model once over the dataset to extract activations.
    Returns a TensorDataset containing these activations.
    """
    activations_list = []
    base_model.eval()

    print("Pre-computing activations...")
    with torch.no_grad():
        for X, _ in tqdm(dataloader, desc="Extracting"):

            # Get activations
            acts = base_model.retrieve_split_layer_activations(X, split_layer)

            # Move back to CPU to store in RAM (prevents GPU OOM)
            activations_list.append(acts.cpu())

    # Concatenate all batches and wrap in a dataset
    all_activations = torch.cat(activations_list)
    return TensorDataset(all_activations)