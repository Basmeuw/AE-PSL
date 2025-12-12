from torch.utils.data import random_split

import available_datasets
from available_datasets import DistributedDataset


# Handles the validation set logic
# For AEs, we don't use the distributedDataset class, hence we do the splitting manually here
def get_dataloaders_from_datasets_AE(train_dataset, test_dataset, validation_mode, validation_split, batch_size, num_workers, dataloader_class, collate_fn=None):
    val_dataloader = None

    # Using a validation set, so create a validation dataloader
    if validation_mode != 'none':
        # move to utils later
        train_size = int((1.0 - validation_split) * len(train_dataset))
        train_dataset, val_dataset = random_split(train_dataset,
                                                            [train_size, len(train_dataset) - train_size])
        val_dataloader = dataloader_class(val_dataset, batch_size=batch_size,
                                         shuffle=False, pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)

    # Don't need to use the custom distributed dataset class here, as we are using activations only.
    train_dataloader = dataloader_class(train_dataset, batch_size=batch_size,
                                       shuffle=True, pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)
    test_dataloader = dataloader_class(test_dataset, batch_size=batch_size,
                                      shuffle=False,
                                      pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)
    print(len(train_dataloader), len(val_dataloader), len(test_dataloader))
    return train_dataloader, val_dataloader, test_dataloader

# Since we use distributedDataset, we should pass the predetermined validation dataset here
def get_dataloaders_from_datasets(train_dataset, validation_dataset, test_dataset, validation_mode, batch_size, num_workers, dataloader_class, small_test_run, collate_fn=None):

    if small_test_run:
        train_dataset = available_datasets.Subset(train_dataset, range(0, len(train_dataset) // 20))
        validation_dataset = available_datasets.Subset(validation_dataset, range(0, len(validation_dataset) // 20))
        test_dataset = available_datasets.Subset(test_dataset, range(0, len(test_dataset) // 20))

    # Using a validation set, so create a validation dataloader
    if validation_mode != 'none':
        val_dataloader = dataloader_class(validation_dataset, batch_size=batch_size,
                                         shuffle=False, pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)
    else:
        val_dataloader = None

    # Don't need to use the custom distributed dataset class here, as we are using activations only.
    train_dataloader = dataloader_class(train_dataset, batch_size=batch_size,
                                       shuffle=True, pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)
    test_dataloader = dataloader_class(test_dataset, batch_size=batch_size,
                                      shuffle=False,
                                      pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)
    print(len(train_dataloader), len(val_dataloader), len(test_dataloader))
    return train_dataloader, val_dataloader, test_dataloader

# each client gets it's own client dataloader, with a partition of the train dataset
def get_distributed_dataloaders_from_datasets(train_dataset, validation_dataset, test_dataset, validation_mode, mini_batch_size, total_batch_size, num_workers, dataloader_class, nr_of_clients, small_test_run, collate_fn=None):

    val_dataloader = None

    client_dataloaders = dict()

    if small_test_run:
        validation_dataset = available_datasets.Subset(validation_dataset, range(0, len(validation_dataset) // 20))
        test_dataset = available_datasets.Subset(test_dataset, range(0, len(test_dataset) // 20))

    for client_id in range(nr_of_clients):
        client_train_data = train_dataset.load_partition(partition_id=client_id)

        if small_test_run:
            client_train_data = available_datasets.Subset(client_train_data, range(0, len(train_dataset) // 20))

        client_train_dataloader = dataloader_class(client_train_data, batch_size=mini_batch_size, shuffle=True,
                                                      pin_memory=True, num_workers=num_workers, collate_fn=collate_fn,
                                                              drop_last=False)
        client_dataloaders[client_id] = client_train_dataloader


    if validation_mode != 'none':
        val_dataloader = dataloader_class(validation_dataset, batch_size=total_batch_size,
                                      shuffle=False, pin_memory=True, num_workers=num_workers,
                                      collate_fn=collate_fn)


    test_dataloader = dataloader_class(test_dataset, batch_size=total_batch_size,
                                       shuffle=False,
                                       pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)

    return client_dataloaders, val_dataloader, test_dataloader