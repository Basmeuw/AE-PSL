from argparse import Namespace

# Minimal manual Namespace (only works if you include every attribute that will be read)
minimal_args = Namespace(
    torch_data_dir='../../shared_data/datasets',
    pre_processors_cache_dir='../../shared_data/preprocessors',
    tokenizer_weights_cache_dir='../../shared_data/tokenizers',
    model_weights_dir='../../shared_data/model_checkpoints',
    ae_weights_dir='../../shared_data/model_checkpoints',
    vit_base='None',

    dataset='cifar10',
    model='vit',
    fusion_type='default',
    trainable_params_key='default',
    use_pre_layer_norm=True,
    use_post_layer_norm=False,
    use_large_encoder=False,
    freeze_encoder=False,
    nr_of_last_encoder_blocks_to_finetune=-1,
    use_clip_encoder_for_text_embeddings=True,
    split_layer=0,
    use_lora=True,
    lora_rank=16,
    lora_alpha=32,

    ae_type='identity',
    ae_latent_dim=384,
    ae_use_existing=False,
    ae_pretrain_dataset='cifar100',
    ae_pretrain_dataset_fraction=1.0,
    ae_pretrain_epochs=2,
    ae_pretrain_batch_size=256,
    ae_pretrain_start_lr=1e-4,
    ae_pretrain_optimizer='adam',
    ae_pretrain_scheduler='step',
    ae_pretrain_scheduler_step_size=5,
    ae_pretrain_loss_fn='mse',
    ae_specific_weights_path=None,
    ae_save_final_weights=True,

    random_seed=2024,
    small_test_run=False,
    batch_size=128,
    num_workers=1,
    nr_of_epochs=3,
    start_lr=1e-4,
    scheduler_step_size=5,
    save_model_after_each_epoch=False,
    save_final_model=True,
    save_file_name=None,
    gpu_id=None,
)

search_space = {
    "seed"
}