import copy

import torch
from torch import nn

from models.auto_encoder import IdentityAE
from models.vision_transformer.base.ae_vision_transformer import AEVisionTransformer

from models.meta_transformer.base.data2seq import InputModality
from utils.mpsl_utils import client_model_requires_any_grad

centralized_base_model = None

def _initialize_base_model(auto_encoder: IdentityAE, split_layer: int, use_lora: bool, lora_rank: int, lora_alpha: int, num_classes: int, device):
    global centralized_base_model

    if centralized_base_model is None:
        centralized_base_model = CentralizedModel(
            auto_encoder=auto_encoder,
            split_layer=split_layer,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            num_classes=num_classes,
            device=device,
        )


    return centralized_base_model

def get_centralized_model(auto_encoder: IdentityAE, split_layer: int, use_lora: bool, lora_rank: int, lora_alpha: int, num_classes: int, device):
    _initialize_base_model(auto_encoder, split_layer, use_lora, lora_rank, lora_alpha, num_classes, device)
    print("centralized_base_model initialized as" , type(centralized_base_model))
    return centralized_base_model

def get_split_model(auto_encoder: IdentityAE, split_layer: int, use_lora: bool, lora_rank: int, lora_alpha: int, num_classes: int, device):
    _initialize_base_model(auto_encoder, split_layer, use_lora, lora_rank, lora_alpha, num_classes, device)

    _client_model = ClientModel(device=device)

    return _client_model, ServerModel(device), client_model_requires_any_grad(_client_model)


class CentralizedModel(AEVisionTransformer):
    def __init__(self, auto_encoder: IdentityAE, split_layer: int, use_lora: bool, lora_rank: int, lora_alpha: int, num_classes: int, device):
        super().__init__(
            auto_encoder=auto_encoder,
            split_layer=split_layer,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            num_classes=num_classes,
            device=device,
        )

        self.device = device

    def forward(self, x):
        return self.forward_full(x)

    # def switch_to_device(self, device):
    #     self.to(device)
    #     return self

class ClientModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.split_layer = centralized_base_model.split_layer

        # --- Deepcopy components from Centralized Model ---
        # 1. Input Processing
        self.patch_size = centralized_base_model.vit.patch_size
        self.hidden_dim = centralized_base_model.vit.hidden_dim
        self.conv_proj = copy.deepcopy(centralized_base_model.vit.conv_proj)

        # 2. Embeddings & Tokens
        self.class_token = copy.deepcopy(centralized_base_model.vit.class_token)
        self.pos_embedding = copy.deepcopy(centralized_base_model.vit.encoder.pos_embedding)
        self.dropout = copy.deepcopy(centralized_base_model.vit.encoder.dropout)

        # 3. Transformer Blocks (up to split)
        # Note: centralized_model.vit.encoder.layers has [Blocks ... AE ... Blocks]
        # We need everything BEFORE the AE.
        client_layers = []
        for i in range(self.split_layer):
            client_layers.append(centralized_base_model.vit.encoder.layers[i])
        self.blocks = copy.deepcopy(nn.Sequential(*client_layers))

        # 4. AE Encoder
        # The AE is located exactly at the split_layer index
        ae_module = centralized_base_model.vit.encoder.layers[self.split_layer]
        self.ae_encoder = copy.deepcopy(ae_module.encoder_net)

        self.to(device)

    def forward(self, x):
        # 1. Patch Embedding Logic
        x = x[InputModality.IMAGE]

        n, c, h, w = x.shape
        p = self.patch_size
        n_h = h // p
        n_w = w // p

        x = self.conv_proj(x)
        x = x.reshape(n, self.hidden_dim, n_h * n_w)
        x = x.permute(0, 2, 1)

        # 2. Add Tokens & Position Embeddings
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)

        # 3. Transformer Blocks
        x = self.blocks(x)

        # 4. AE Encoder
        x = self.ae_encoder(x)

        return x


class ServerModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        # Server model is only replicated a single time, so we don't need to deepcopy

    def forward(self, x):
        return centralized_base_model.forward_server(x)