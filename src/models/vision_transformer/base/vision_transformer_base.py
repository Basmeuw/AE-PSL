import torchvision
from torch import nn

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


        self.vit.to(device)

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

    def _apply_lora_to_encoder(self, rank, alpha):
        """Recursively swaps Linear layers for LoRALinear in the encoder."""

        def replace_linear_recursion(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    setattr(module, name, LoRALinear(child, rank, alpha))
                else:
                    replace_linear_recursion(child)

        replace_linear_recursion(self.vit.encoder)