import torch
import torchvision
from torch import nn

from models import InputModality
from models.lora_layer import LoRALinear


class VisionTransformerBase(nn.Module):
    def __init__(self, vit_type: str, use_lora: bool, lora_rank: int, lora_alpha: int, num_classes: int, device):
        super().__init__()
        self.use_lora = use_lora

        if vit_type == "vit_b_16":
            print(f"Initializing ViT-B/16 (Mode: {'LoRA' if use_lora else 'Full Fine-Tuning'})...")
            weights = torchvision.models.ViT_B_16_Weights.DEFAULT
            self.vit = torchvision.models.vit_b_16(weights=weights)
        elif vit_type == "vit_b_32":
            print(f"Initializing ViT-B/32 (Mode: {'LoRA' if use_lora else 'Full Fine-Tuning'})...")
            weights = torchvision.models.ViT_B_32_Weights.DEFAULT
            self.vit = torchvision.models.vit_b_32(weights=weights)
        else:
            raise ValueError(f"Unsupported ViT type: {vit_type}")



        self.device = device

        # Replace with new head for the desired number of classes
        self.vit.heads = nn.Linear(self.vit.heads[0].in_features, num_classes)

        if self.use_lora:
            # MODE A: LoRA Adaptation
            # 1. Freeze EVERYTHING first
            for param in self.vit.parameters():
                param.requires_grad = False

            # 2. Swap Encoder Linear layers to LoRALinear (adds trainable params)
            self._apply_lora_to_encoder(rank=lora_rank, alpha=lora_alpha)

            # 3. Unfreeze Head (Standard practice)
            for param in self.vit.heads.parameters():
                param.requires_grad = True

        else:
            # MODE B: Regular Fine-Tuning
            # Ensure everything is trainable (default behavior, but being explicit is safe)
            for param in self.vit.parameters():
                param.requires_grad = True

        self.vit.to(device)

    def _apply_lora_to_encoder(self, rank, alpha):
        """Recursively swaps Linear layers for LoRALinear in the encoder."""

        def replace_linear_recursion(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    setattr(module, name, LoRALinear(child, rank, alpha))
                else:
                    replace_linear_recursion(child)

        replace_linear_recursion(self.vit.encoder)

    def retrieve_split_layer_activations(self, x, split_layer: int):

        with torch.no_grad():
            x = x[InputModality.IMAGE]
            x = x.to(self.device)
            # 1. Patch Embedding Logic
            n, c, h, w = x.shape
            p = self.vit.patch_size
            torch._assert(h == self.vit.image_size, f"Wrong image height! Expected {self.vit.image_size} but got {h}!")
            torch._assert(w == self.vit.image_size, f"Wrong image width! Expected {self.vit.image_size} but got {w}!")
            n_h = h // p
            n_w = w // p

            # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
            x = self.vit.conv_proj(x)
            # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
            x = x.reshape(n, self.vit.hidden_dim, n_h * n_w)

            # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
            # The self attention layer expects inputs in the format (N, S, E)
            # where S is the source sequence length, N is the batch size, E is the
            # embedding dimension
            x = x.permute(0, 2, 1)

            # 2. Add Tokens & Position Embeddings
            batch_class_token = self.vit.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")

            x = x + self.vit.encoder.pos_embedding
            x = self.vit.encoder.dropout(x)

            # 3. Forward Pass through Transformer Encoder up to split_layer
            for i in range(split_layer):
                x = self.vit.encoder.layers[i](x)


        return x

    def get_hidden_dim(self):
        return self.vit.hidden_dim