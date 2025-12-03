from argparse import Namespace
from enum import Enum
from typing import Any

import torch

from models.meta_transformer.base.data2seq import InputModality
from models.meta_transformer.implementations.unimodal.image_classification.models import get_split_model as get_meta_cifar_100_split_model, get_centralized_model as get_meta_cifar_100_centralized_model

from models.meta_transformer.implementations.multimodal.vqa.models import get_split_model as get_meta_vqa_split_model, get_centralized_model as get_meta_vqa_centralized_model
from models.meta_transformer.implementations.multimodal.vqa.models import get_federated_model as get_meta_vqa_federated_model

from models.auto_encoder import IdentityAE
from models.meta_transformer.implementations.multimodal.image_text_retrieval.models import get_split_model as get_meta_image_text_retrieval_split_model, get_centralized_model as get_meta_image_text_retrieval_centralized_model
from models.meta_transformer.implementations.multimodal.image_text_retrieval.models import get_federated_model as get_meta_image_text_retrieval_federated_model
from models.meta_transformer.implementations.multimodal.action_recognition.models import get_split_model as get_meta_action_recognition_split_model, get_centralized_model as get_meta_action_recognition_centralized_model
from models.meta_transformer.implementations.multimodal.action_recognition.models import get_federated_model as get_meta_action_recognition_federated_model
from models.meta_transformer.implementations.multimodal.emotion_recognition.models import get_split_model as get_meta_meld_split_model, get_centralized_model as get_meta_emotion_recognition_centralized_model
from models.meta_transformer.implementations.multimodal.emotion_recognition.models import get_federated_model as get_meta_meld_federated_model

from trainers.implementations.classification.centralized_trainer import CentralizedTrainer as classification_centralized_trainer
from trainers.implementations.experiment_trainer import ExperimentTrainer
from trainers.implementations.image_text_retrieval.centralized_trainer import CentralizedTrainer as image_text_retrieval_centralized_trainer

from trainers.implementations.classification.mpsl_trainer import MPSLTrainer as classification_mpsl_trainer
from trainers.implementations.image_text_retrieval.mpsl_trainer import MPSLTrainer as image_text_retrieval_mpsl_trainer

from trainers.implementations.classification.fl_trainer import FLTrainer as classification_fl_trainer
from trainers.implementations.image_text_retrieval.fl_trainer import FLTrainer as image_text_retrieval_fl_trainer




class SupportedModel(Enum):

    META_TRANSFORMER = 'meta_transformer'
    VIT = 'vit'

# Choose between models
def get_centralized_model_and_trainer(global_args: Namespace, device: torch.device, auto_encoder: IdentityAE = None) -> (Any, ExperimentTrainer):
    if global_args.model.upper() not in SupportedModel.__members__.keys():
        raise NotImplementedError('Chosen model is currently not supported.')

    model = SupportedModel[global_args.model.upper()]

    if model is SupportedModel.META_TRANSFORMER:
        return get_centralized_model_and_trainer_meta_transformer(global_args, device)
    elif model is SupportedModel.VIT:
        if auto_encoder is None:
            raise ValueError("Auto-encoder instance must be provided to get_centralized_model_and_trainer for VIT model.")
        return get_centralized_model_and_trainer_vit(global_args, auto_encoder, device)
    else:
        raise NotImplementedError('Chosen model is currently not supported.')

# Choose between models TODO
def get_split_model_pair_and_trainer(global_args: Namespace, device: torch.device, auto_encoder: IdentityAE = None) -> (
Any, ExperimentTrainer):
    if global_args.model.upper() not in SupportedModel.__members__.keys():
        raise NotImplementedError('Chosen model is currently not supported.')

    model = SupportedModel[global_args.model.upper()]
    if model is SupportedModel.META_TRANSFORMER:
        return get_split_model_pair_and_trainer_meta_transformer(global_args, device)
    elif model is SupportedModel.VIT:
        if auto_encoder is None:
            raise ValueError("Auto-encoder instance must be provided to get_centralized_model_and_trainer for VIT model.")
        return get_split_model_and_trainer_vit(global_args, auto_encoder, device)


# Choose between models
def get_federated_model_and_trainer(global_args: Namespace, device: torch.device, auto_encoder: IdentityAE = None) -> (
Any, ExperimentTrainer):
    if global_args.model.upper() not in SupportedModel.__members__.keys():
        raise NotImplementedError('Chosen model is currently not supported.')

    model = SupportedModel[global_args.model.upper()]
    if model is SupportedModel.META_TRANSFORMER:
        return get_federated_model_and_trainer_meta_transformer(global_args, device)
    elif model is SupportedModel.VIT:
        raise NotImplementedError('VIT is currently not supported for FL.')

def get_centralized_model_and_trainer_vit(global_args: Namespace, auto_encoder: IdentityAE, device: torch.device) -> (Any, ExperimentTrainer):
    dataset = global_args.dataset
    if dataset == 'cifar100':
        from models.vision_transformer.implementations.unimodal.image_classification.models import get_centralized_model as get_centralized_model_vit
        return get_centralized_model_vit(
            auto_encoder=auto_encoder,
            split_layer=global_args.split_layer,
            use_lora=global_args.use_lora,
            lora_rank=global_args.lora_rank,
            lora_alpha=global_args.lora_alpha,
            num_classes=100,
            device=device
        ), classification_centralized_trainer()
    else:
        raise NotImplementedError(f'Chosen model\'s dataset {dataset} is currently not supported.')

def get_split_model_and_trainer_vit(global_args: Namespace, auto_encoder: IdentityAE, device: torch.device) -> (Any, ExperimentTrainer):
    dataset = global_args.dataset
    if dataset == 'cifar100':
        from src.models.vision_transformer.implementations.unimodal.image_classification.models import get_split_model as get_split_model_vit
        return get_split_model_vit(
            auto_encoder=auto_encoder,
            split_layer=global_args.split_layer,
            use_lora=global_args.use_lora,
            lora_rank=global_args.lora_rank,
            lora_alpha=global_args.lora_alpha,
            num_classes=100,
            device=device
        ), classification_mpsl_trainer()
    else:
        raise NotImplementedError(f'Chosen model\'s dataset {dataset} is currently not supported.')

def get_federated_model_and_trainer_meta_transformer(global_args: Namespace, device: torch.device) -> (Any, ExperimentTrainer):
    dataset = global_args.dataset

    if global_args.model.upper() not in SupportedModel.__members__.keys():
        raise NotImplementedError('Chosen model is currently not supported.')

    if dataset == 'cifar100':
        if global_args.use_adapter_approach:
            raise NotImplementedError()

        return get_meta_cifar_100_centralized_model(
            use_pre_layer_norm=global_args.use_pre_layer_norm,
            use_post_layer_norm=global_args.use_post_layer_norm,
            use_large_encoder=global_args.use_large_encoder,
            freeze_encoder=global_args.freeze_encoder,
            nr_of_encoder_blocks_to_finetune=global_args.nr_of_last_encoder_blocks_to_finetune,
            trainable_params_key=global_args.trainable_params_key,
            device=device
        ), classification_fl_trainer()

    elif dataset == 'coco-qa':
        from available_datasets.multimodal.vqa.coco_qa import NR_OF_CLASSES

        return get_meta_vqa_federated_model(
            use_pre_layer_norm=global_args.use_pre_layer_norm,
            use_post_layer_norm=global_args.use_post_layer_norm,
            use_large_encoder=global_args.use_large_encoder,
            freeze_encoder=global_args.freeze_encoder,
            nr_of_encoder_blocks_to_finetune=global_args.nr_of_last_encoder_blocks_to_finetune,
            trainable_params_key=global_args.trainable_params_key,
            use_clip_encoder_for_text_embeddings=global_args.use_clip_encoder_for_text_embeddings,
            device=device,
            num_classes=NR_OF_CLASSES,
            fusion_type=global_args.fusion_type,
            use_adapter_approach=global_args.use_adapter_approach,
            include_image_adapter=global_args.include_image_adapter,
            include_text_adapter=global_args.include_text_adapter,
            include_unified_adapter=global_args.include_unified_adapter
        ), classification_fl_trainer()
    elif dataset == 't4sa':
        from available_datasets.multimodal.sentiment_analysis.t4sa import NR_OF_CLASSES

        return get_meta_vqa_federated_model(
            use_pre_layer_norm=global_args.use_pre_layer_norm,
            use_post_layer_norm=global_args.use_post_layer_norm,
            use_large_encoder=global_args.use_large_encoder,
            freeze_encoder=global_args.freeze_encoder,
            nr_of_encoder_blocks_to_finetune=global_args.nr_of_last_encoder_blocks_to_finetune,
            trainable_params_key=global_args.trainable_params_key,
            use_clip_encoder_for_text_embeddings=global_args.use_clip_encoder_for_text_embeddings,
            device=device,
            num_classes=NR_OF_CLASSES,
            fusion_type=global_args.fusion_type,
            use_adapter_approach=global_args.use_adapter_approach,
            include_image_adapter=global_args.include_image_adapter,
            include_text_adapter=global_args.include_text_adapter,
            include_unified_adapter=global_args.include_unified_adapter
        ), classification_fl_trainer()

    elif dataset == 'flickr30k':
        from available_datasets.multimodal.image_text_retrieval.flickr30k import NR_OF_CAPTIONS_PER_IMAGE, TEST_SET_LEN

        return get_meta_image_text_retrieval_federated_model(
            use_pre_layer_norm=global_args.use_pre_layer_norm,
            use_post_layer_norm=global_args.use_post_layer_norm,
            use_large_encoder=global_args.use_large_encoder,
            freeze_encoder=global_args.freeze_encoder,
            nr_of_encoder_blocks_to_finetune=global_args.nr_of_last_encoder_blocks_to_finetune,
            trainable_params_key=global_args.trainable_params_key,
            use_clip_encoder_for_text_embeddings=global_args.use_clip_encoder_for_text_embeddings,
            device=device,
            use_adapter_approach=global_args.use_adapter_approach,
            include_image_adapter=global_args.include_image_adapter,
            include_text_adapter=global_args.include_text_adapter,
            include_unified_adapter=global_args.include_unified_adapter
        ), image_text_retrieval_fl_trainer(NR_OF_CAPTIONS_PER_IMAGE, TEST_SET_LEN, global_args.batch_size, move_dist_matrix_to_cpu=global_args.move_dist_matrix_to_cpu)
    elif dataset == 'coco-retrieval':
        from available_datasets.multimodal.image_text_retrieval.coco_retrieval import NR_OF_CAPTIONS_PER_IMAGE, TEST_SET_LEN

        return get_meta_image_text_retrieval_federated_model(
            use_pre_layer_norm=global_args.use_pre_layer_norm,
            use_post_layer_norm=global_args.use_post_layer_norm,
            use_large_encoder=global_args.use_large_encoder,
            freeze_encoder=global_args.freeze_encoder,
            nr_of_encoder_blocks_to_finetune=global_args.nr_of_last_encoder_blocks_to_finetune,
            trainable_params_key=global_args.trainable_params_key,
            use_clip_encoder_for_text_embeddings=global_args.use_clip_encoder_for_text_embeddings,
            device=device,
            use_adapter_approach=global_args.use_adapter_approach,
            include_image_adapter=global_args.include_image_adapter,
            include_text_adapter=global_args.include_text_adapter,
            include_unified_adapter=global_args.include_unified_adapter
        ), image_text_retrieval_fl_trainer(NR_OF_CAPTIONS_PER_IMAGE, TEST_SET_LEN, global_args.batch_size, move_dist_matrix_to_cpu=global_args.move_dist_matrix_to_cpu)
    elif dataset == 'ucf101':
        from available_datasets.multimodal.action_recognition.ucf_101 import NR_OF_CLASSES

        return get_meta_action_recognition_federated_model(
            use_pre_layer_norm=global_args.use_pre_layer_norm,
            use_post_layer_norm=global_args.use_post_layer_norm,
            use_large_encoder=global_args.use_large_encoder,
            freeze_encoder=global_args.freeze_encoder,
            nr_of_encoder_blocks_to_finetune=global_args.nr_of_last_encoder_blocks_to_finetune,
            trainable_params_key=global_args.trainable_params_key,
            device=device,
            fusion_type=global_args.fusion_type,
            audio_time_dimension=global_args.audio_time_dimension,
            num_classes=NR_OF_CLASSES,
            use_adapter_approach=global_args.use_adapter_approach,
            include_image_adapter=global_args.include_image_adapter,
            include_audio_adapter=global_args.include_audio_adapter,
            include_unified_adapter=global_args.include_unified_adapter
        ), classification_fl_trainer()
    elif dataset == 'kinetics-sounds':
        from available_datasets.multimodal.action_recognition.kinetics_sounds import NR_OF_CLASSES

        return get_meta_action_recognition_federated_model(
            use_pre_layer_norm=global_args.use_pre_layer_norm,
            use_post_layer_norm=global_args.use_post_layer_norm,
            use_large_encoder=global_args.use_large_encoder,
            freeze_encoder=global_args.freeze_encoder,
            nr_of_encoder_blocks_to_finetune=global_args.nr_of_last_encoder_blocks_to_finetune,
            trainable_params_key=global_args.trainable_params_key,
            device=device,
            fusion_type=global_args.fusion_type,
            audio_time_dimension=global_args.audio_time_dimension,
            num_classes=NR_OF_CLASSES,
            use_adapter_approach=global_args.use_adapter_approach,
            include_image_adapter=global_args.include_image_adapter,
            include_audio_adapter=global_args.include_audio_adapter,
            include_unified_adapter=global_args.include_unified_adapter
        ), classification_fl_trainer()

    elif dataset == 'meld':
        from available_datasets.multimodal.emotion_recognition.meld import NR_OF_CLASSES

        return get_meta_meld_federated_model(
            use_pre_layer_norm=global_args.use_pre_layer_norm,
            use_post_layer_norm=global_args.use_post_layer_norm,
            use_large_encoder=global_args.use_large_encoder,
            freeze_encoder=global_args.freeze_encoder,
            nr_of_encoder_blocks_to_finetune=global_args.nr_of_last_encoder_blocks_to_finetune,
            trainable_params_key=global_args.trainable_params_key,
            use_clip_encoder_for_text_embeddings=global_args.use_clip_encoder_for_text_embeddings,
            device=device,
            fusion_type=global_args.fusion_type,
            audio_time_dimension=global_args.audio_time_dimension,
            num_classes=NR_OF_CLASSES
        ), classification_fl_trainer()
    else:
        raise NotImplementedError('Chosen model\'s dataset implementation is currently not supported.')


def get_split_model_pair_and_trainer_meta_transformer(global_args: Namespace, device: torch.device) -> ((Any, Any, bool), ExperimentTrainer):
    dataset = global_args.dataset

    if global_args.model.upper() not in SupportedModel.__members__.keys():
        raise NotImplementedError('Chosen model is currently not supported.')

    if dataset == 'cifar100':
        return (get_meta_cifar_100_split_model(
            use_pre_layer_norm=global_args.use_pre_layer_norm,
            use_post_layer_norm=global_args.use_post_layer_norm,
            use_large_encoder=global_args.use_large_encoder,
            freeze_encoder=global_args.freeze_encoder,
            nr_of_encoder_blocks_to_finetune=global_args.nr_of_last_encoder_blocks_to_finetune,
            trainable_params_key=global_args.trainable_params_key,
            device=device
        )), classification_mpsl_trainer(global_args.fusion_type, [InputModality.IMAGE])

    elif dataset == 'coco-qa':
        from available_datasets.multimodal.vqa.coco_qa import NR_OF_CLASSES

        client_model, server_model, client_model_requires_any_grad = get_meta_vqa_split_model(
            use_pre_layer_norm=global_args.use_pre_layer_norm,
            use_post_layer_norm=global_args.use_post_layer_norm,
            use_large_encoder=global_args.use_large_encoder,
            freeze_encoder=global_args.freeze_encoder,
            nr_of_encoder_blocks_to_finetune=global_args.nr_of_last_encoder_blocks_to_finetune,
            trainable_params_key=global_args.trainable_params_key,
            use_clip_encoder_for_text_embeddings=global_args.use_clip_encoder_for_text_embeddings,
            device=device,
            num_classes=NR_OF_CLASSES,
            fusion_type=global_args.fusion_type
        )

        # Note usage of client_model.fusion_type instead of args.fusion_type; The latter might still be 'default' that has yet to be parsed, whereas the former will always be parsed.
        return (client_model, server_model, client_model_requires_any_grad), classification_mpsl_trainer(client_model.fusion_type, [InputModality.IMAGE, InputModality.TEXT])
    elif dataset == 't4sa':
        from available_datasets.multimodal.sentiment_analysis.t4sa import NR_OF_CLASSES

        client_model, server_model, client_model_requires_any_grad = get_meta_vqa_split_model(
            use_pre_layer_norm=global_args.use_pre_layer_norm,
            use_post_layer_norm=global_args.use_post_layer_norm,
            use_large_encoder=global_args.use_large_encoder,
            freeze_encoder=global_args.freeze_encoder,
            nr_of_encoder_blocks_to_finetune=global_args.nr_of_last_encoder_blocks_to_finetune,
            trainable_params_key=global_args.trainable_params_key,
            use_clip_encoder_for_text_embeddings=global_args.use_clip_encoder_for_text_embeddings,
            device=device,
            num_classes=NR_OF_CLASSES,
            fusion_type=global_args.fusion_type
        )

        # Note usage of client_model.fusion_type instead of args.fusion_type; The latter might still be 'default' that has yet to be parsed, whereas the former will always be parsed.
        return (client_model, server_model, client_model_requires_any_grad), classification_mpsl_trainer(client_model.fusion_type, [InputModality.IMAGE, InputModality.TEXT])

    elif dataset == 'flickr30k':
        from available_datasets.multimodal.image_text_retrieval.flickr30k import NR_OF_CAPTIONS_PER_IMAGE, TEST_SET_LEN

        return get_meta_image_text_retrieval_split_model(
            use_pre_layer_norm=global_args.use_pre_layer_norm,
            use_post_layer_norm=global_args.use_post_layer_norm,
            use_large_encoder=global_args.use_large_encoder,
            freeze_encoder=global_args.freeze_encoder,
            nr_of_encoder_blocks_to_finetune=global_args.nr_of_last_encoder_blocks_to_finetune,
            trainable_params_key=global_args.trainable_params_key,
            use_clip_encoder_for_text_embeddings=global_args.use_clip_encoder_for_text_embeddings,
            device=device
        ), image_text_retrieval_mpsl_trainer(NR_OF_CAPTIONS_PER_IMAGE, TEST_SET_LEN, global_args.batch_size)
    elif dataset == 'coco-retrieval':
        from available_datasets.multimodal.image_text_retrieval.coco_retrieval import NR_OF_CAPTIONS_PER_IMAGE, TEST_SET_LEN

        return get_meta_image_text_retrieval_split_model(
            use_pre_layer_norm=global_args.use_pre_layer_norm,
            use_post_layer_norm=global_args.use_post_layer_norm,
            use_large_encoder=global_args.use_large_encoder,
            freeze_encoder=global_args.freeze_encoder,
            nr_of_encoder_blocks_to_finetune=global_args.nr_of_last_encoder_blocks_to_finetune,
            trainable_params_key=global_args.trainable_params_key,
            use_clip_encoder_for_text_embeddings=global_args.use_clip_encoder_for_text_embeddings,
            device=device
        ), image_text_retrieval_mpsl_trainer(NR_OF_CAPTIONS_PER_IMAGE, TEST_SET_LEN, global_args.batch_size)

    elif dataset == 'ucf101':
        from available_datasets.multimodal.action_recognition.ucf_101 import NR_OF_CLASSES

        client_model, server_model, client_model_requires_any_grad = get_meta_action_recognition_split_model(
            use_pre_layer_norm=global_args.use_pre_layer_norm,
            use_post_layer_norm=global_args.use_post_layer_norm,
            use_large_encoder=global_args.use_large_encoder,
            freeze_encoder=global_args.freeze_encoder,
            nr_of_encoder_blocks_to_finetune=global_args.nr_of_last_encoder_blocks_to_finetune,
            trainable_params_key=global_args.trainable_params_key,
            device=device,
            fusion_type=global_args.fusion_type,
            audio_time_dimension=global_args.audio_time_dimension,
            num_classes=NR_OF_CLASSES
        )
        # Note usage of client_model.fusion_type instead of args.fusion_type; The latter might still be 'default' that has yet to be parsed, whereas the former will always be parsed.
        return (client_model, server_model, client_model_requires_any_grad), classification_mpsl_trainer(client_model.fusion_type, [InputModality.IMAGE, InputModality.AUDIO])
    elif dataset == 'kinetics-sounds':
        from available_datasets.multimodal.action_recognition.kinetics_sounds import NR_OF_CLASSES

        client_model, server_model, client_model_requires_any_grad = get_meta_action_recognition_split_model(
            use_pre_layer_norm=global_args.use_pre_layer_norm,
            use_post_layer_norm=global_args.use_post_layer_norm,
            use_large_encoder=global_args.use_large_encoder,
            freeze_encoder=global_args.freeze_encoder,
            nr_of_encoder_blocks_to_finetune=global_args.nr_of_last_encoder_blocks_to_finetune,
            trainable_params_key=global_args.trainable_params_key,
            device=device,
            fusion_type=global_args.fusion_type,
            audio_time_dimension=global_args.audio_time_dimension,
            num_classes=NR_OF_CLASSES
        )
        # Note usage of client_model.fusion_type instead of args.fusion_type; The latter might still be 'default' that has yet to be parsed, whereas the former will always be parsed.
        return (client_model, server_model, client_model_requires_any_grad), classification_mpsl_trainer(client_model.fusion_type, [InputModality.IMAGE, InputModality.AUDIO])
    elif dataset == 'meld':
        from available_datasets.multimodal.emotion_recognition.meld import NR_OF_CLASSES

        client_model, server_model, client_model_requires_any_grad = get_meta_meld_split_model(
            use_pre_layer_norm=global_args.use_pre_layer_norm,
            use_post_layer_norm=global_args.use_post_layer_norm,
            use_large_encoder=global_args.use_large_encoder,
            freeze_encoder=global_args.freeze_encoder,
            nr_of_encoder_blocks_to_finetune=global_args.nr_of_last_encoder_blocks_to_finetune,
            trainable_params_key=global_args.trainable_params_key,
            use_clip_encoder_for_text_embeddings=global_args.use_clip_encoder_for_text_embeddings,
            device=device,
            fusion_type=global_args.fusion_type,
            audio_time_dimension=global_args.audio_time_dimension,
            num_classes=NR_OF_CLASSES
        )

        # Note usage of client_model.fusion_type instead of args.fusion_type; The latter might still be 'default' that has yet to be parsed, whereas the former will always be parsed.
        return (client_model, server_model, client_model_requires_any_grad), classification_mpsl_trainer(client_model.fusion_type, [InputModality.TEXT, InputModality.AUDIO])
    else:
        raise NotImplementedError('Chosen model\'s dataset implementation is currently not supported.')


def get_centralized_model_and_trainer_meta_transformer(global_args: Namespace, device: torch.device) -> (Any, ExperimentTrainer):
    dataset = global_args.dataset

    if global_args.model.upper() not in SupportedModel.__members__.keys():
        raise NotImplementedError('Chosen model is currently not supported.')

    if dataset == 'cifar100':
        return get_meta_cifar_100_centralized_model(
            use_pre_layer_norm=global_args.use_pre_layer_norm,
            use_post_layer_norm=global_args.use_post_layer_norm,
            use_large_encoder=global_args.use_large_encoder,
            freeze_encoder=global_args.freeze_encoder,
            nr_of_encoder_blocks_to_finetune=global_args.nr_of_last_encoder_blocks_to_finetune,
            trainable_params_key=global_args.trainable_params_key,
            device=device
        ), classification_centralized_trainer()

    elif dataset == 'coco-qa':
        from available_datasets.multimodal.vqa.coco_qa import NR_OF_CLASSES

        return get_meta_vqa_centralized_model(
            use_pre_layer_norm=global_args.use_pre_layer_norm,
            use_post_layer_norm=global_args.use_post_layer_norm,
            use_large_encoder=global_args.use_large_encoder,
            freeze_encoder=global_args.freeze_encoder,
            nr_of_encoder_blocks_to_finetune=global_args.nr_of_last_encoder_blocks_to_finetune,
            trainable_params_key=global_args.trainable_params_key,
            use_clip_encoder_for_text_embeddings=global_args.use_clip_encoder_for_text_embeddings,
            device=device,
            num_classes=NR_OF_CLASSES,
            fusion_type=global_args.fusion_type,
            use_adapter_approach=global_args.use_adapter_approach,
            include_image_adapter=global_args.include_image_adapter,
            include_text_adapter=global_args.include_text_adapter,
            include_unified_adapter=global_args.include_unified_adapter
        ), classification_centralized_trainer()
    elif dataset == 't4sa':
        from available_datasets.multimodal.sentiment_analysis.t4sa import NR_OF_CLASSES

        return get_meta_vqa_centralized_model(
            use_pre_layer_norm=global_args.use_pre_layer_norm,
            use_post_layer_norm=global_args.use_post_layer_norm,
            use_large_encoder=global_args.use_large_encoder,
            freeze_encoder=global_args.freeze_encoder,
            nr_of_encoder_blocks_to_finetune=global_args.nr_of_last_encoder_blocks_to_finetune,
            trainable_params_key=global_args.trainable_params_key,
            use_clip_encoder_for_text_embeddings=global_args.use_clip_encoder_for_text_embeddings,
            device=device,
            num_classes=NR_OF_CLASSES,
            fusion_type=global_args.fusion_type,
            use_adapter_approach=global_args.use_adapter_approach,
            include_image_adapter=global_args.include_image_adapter,
            include_text_adapter=global_args.include_text_adapter,
            include_unified_adapter=global_args.include_unified_adapter
        ), classification_centralized_trainer()

    elif dataset == 'flickr30k':
        from available_datasets.multimodal.image_text_retrieval.flickr30k import NR_OF_CAPTIONS_PER_IMAGE, TEST_SET_LEN

        return get_meta_image_text_retrieval_centralized_model(
            use_pre_layer_norm=global_args.use_pre_layer_norm,
            use_post_layer_norm=global_args.use_post_layer_norm,
            use_large_encoder=global_args.use_large_encoder,
            freeze_encoder=global_args.freeze_encoder,
            nr_of_encoder_blocks_to_finetune=global_args.nr_of_last_encoder_blocks_to_finetune,
            trainable_params_key=global_args.trainable_params_key,
            use_clip_encoder_for_text_embeddings=global_args.use_clip_encoder_for_text_embeddings,
            device=device,
            use_adapter_approach=global_args.use_adapter_approach,
            include_image_adapter=global_args.include_image_adapter,
            include_text_adapter=global_args.include_text_adapter,
            include_unified_adapter=global_args.include_unified_adapter
        ), image_text_retrieval_centralized_trainer(NR_OF_CAPTIONS_PER_IMAGE, TEST_SET_LEN, global_args.batch_size)
    elif dataset == 'coco-retrieval':
        from available_datasets.multimodal.image_text_retrieval.coco_retrieval import NR_OF_CAPTIONS_PER_IMAGE, TEST_SET_LEN

        return get_meta_image_text_retrieval_centralized_model(
            use_pre_layer_norm=global_args.use_pre_layer_norm,
            use_post_layer_norm=global_args.use_post_layer_norm,
            use_large_encoder=global_args.use_large_encoder,
            freeze_encoder=global_args.freeze_encoder,
            nr_of_encoder_blocks_to_finetune=global_args.nr_of_last_encoder_blocks_to_finetune,
            trainable_params_key=global_args.trainable_params_key,
            use_clip_encoder_for_text_embeddings=global_args.use_clip_encoder_for_text_embeddings,
            device=device,
            use_adapter_approach=global_args.use_adapter_approach,
            include_image_adapter=global_args.include_image_adapter,
            include_text_adapter=global_args.include_text_adapter,
            include_unified_adapter=global_args.include_unified_adapter
        ), image_text_retrieval_centralized_trainer(NR_OF_CAPTIONS_PER_IMAGE, TEST_SET_LEN, global_args.batch_size)
    elif dataset == 'ucf101':
        from available_datasets.multimodal.action_recognition.ucf_101 import NR_OF_CLASSES

        return get_meta_action_recognition_centralized_model(
            use_pre_layer_norm=global_args.use_pre_layer_norm,
            use_post_layer_norm=global_args.use_post_layer_norm,
            use_large_encoder=global_args.use_large_encoder,
            freeze_encoder=global_args.freeze_encoder,
            nr_of_encoder_blocks_to_finetune=global_args.nr_of_last_encoder_blocks_to_finetune,
            trainable_params_key=global_args.trainable_params_key,
            device=device,
            fusion_type=global_args.fusion_type,
            audio_time_dimension=global_args.audio_time_dimension,
            num_classes=NR_OF_CLASSES,
            use_adapter_approach=global_args.use_adapter_approach,
            include_image_adapter=global_args.include_image_adapter,
            include_audio_adapter=global_args.include_audio_adapter,
            include_unified_adapter=global_args.include_unified_adapter
        ), classification_centralized_trainer()
    elif dataset == 'kinetics-sounds':
        from available_datasets.multimodal.action_recognition.kinetics_sounds import NR_OF_CLASSES

        return get_meta_action_recognition_centralized_model(
            use_pre_layer_norm=global_args.use_pre_layer_norm,
            use_post_layer_norm=global_args.use_post_layer_norm,
            use_large_encoder=global_args.use_large_encoder,
            freeze_encoder=global_args.freeze_encoder,
            nr_of_encoder_blocks_to_finetune=global_args.nr_of_last_encoder_blocks_to_finetune,
            trainable_params_key=global_args.trainable_params_key,
            device=device,
            fusion_type=global_args.fusion_type,
            audio_time_dimension=global_args.audio_time_dimension,
            num_classes=NR_OF_CLASSES,
            use_adapter_approach=global_args.use_adapter_approach,
            include_image_adapter=global_args.include_image_adapter,
            include_audio_adapter=global_args.include_audio_adapter,
            include_unified_adapter=global_args.include_unified_adapter
        ), classification_centralized_trainer()

    elif dataset == 'meld':
        from available_datasets.multimodal.emotion_recognition.meld import NR_OF_CLASSES

        return get_meta_emotion_recognition_centralized_model(
            use_pre_layer_norm=global_args.use_pre_layer_norm,
            use_post_layer_norm=global_args.use_post_layer_norm,
            use_large_encoder=global_args.use_large_encoder,
            freeze_encoder=global_args.freeze_encoder,
            nr_of_encoder_blocks_to_finetune=global_args.nr_of_last_encoder_blocks_to_finetune,
            trainable_params_key=global_args.trainable_params_key,
            use_clip_encoder_for_text_embeddings=global_args.use_clip_encoder_for_text_embeddings,
            device=device,
            fusion_type=global_args.fusion_type,
            audio_time_dimension=global_args.audio_time_dimension,
            num_classes=NR_OF_CLASSES,
            include_text_adapter=global_args.include_text_adapter,
            include_audio_adapter=global_args.include_audio_adapter,
            include_unified_adapter=global_args.include_unified_adapter
        ), classification_centralized_trainer()
    else:
        raise NotImplementedError('Chosen model\'s dataset implementation is currently not supported.')
