from torch.utils.data import random_split


# Handles the validation set logic
def get_dataloaders_from_datasets(train_dataset, test_dataset, validation_mode, validation_split, batch_size, num_workers, dataloader_class, collate_fn=None):
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

    return train_dataloader, val_dataloader, test_dataloader