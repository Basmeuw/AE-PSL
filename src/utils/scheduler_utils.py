import torch


def get_optimizer_and_scheduler(model, global_args):
    optimizer = torch.optim.Adam(model.parameters(), lr=global_args['start_lr'])

    return optimizer, get_scheduler(optimizer, global_args)




def get_scheduler(optimizer, global_args):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=global_args['scheduler_step_size'])


def get_ae_pretrain_optimizer_and_scheduler(model, global_args):

    if global_args['ae_pretrain_optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=global_args['ae_pretrain_start_lr'])
    else:
        raise NotImplementedError(f"Unsupported optimizer: {global_args['ae_pretrain_optimizer']}")

    if global_args['ae_pretrain_scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=global_args['ae_pretrain_scheduler_step_size'])
    elif global_args['ae_pretrain_scheduler'] == 'cosine':
        print("Warning, using CosineAnnealingLR default min lr 5e-6")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=global_args['ae_pretrain_epochs'], eta_min=5e-6)
    else:
        raise NotImplementedError(f"Unsupported scheduler: {global_args['ae_pretrain_scheduler']}")
    return optimizer, scheduler