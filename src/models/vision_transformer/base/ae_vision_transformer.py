import torchvision
from torch import nn

from src.models.auto_encoder import AE
from src.models.vision_transformer.base.vision_transformer_base import VisionTransformerBase


class AEVisionTransformer(VisionTransformerBase):
    def __init__(self, auto_encoder: AE, split_layer: int, use_lora: bool = True, lora_rank=4, num_classes=100):
        # 1. Initialize Base (Handles LoRA vs Regular switch)
        super().__init__(use_lora=use_lora, lora_rank=lora_rank)

        # 2. Create AE (Always Standard, Always Trainable)
        hidden_dim = self.vit.hidden_dim

        # 3. Insert AE into the encoder sequence
        encoder_layers = list(self.vit.encoder.layers)

        if split_layer < 0 or split_layer > len(encoder_layers):
            raise ValueError(f"Split layer must be between 0 and {len(encoder_layers)}")

        encoder_layers.insert(split_layer, auto_encoder)
        self.vit.encoder.layers = nn.Sequential(*encoder_layers)

    def forward(self, x):
        return self.vit(x)

    def print_status(self):
        """Helper to inspect the model state."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.parameters())

        mode = "LoRA" if self.use_lora else "Full Fine-Tuning"
        print(f"\n--- Model Status: {mode} ---")
        print(f"Total Params: {all_params:,}")
        print(f"Trainable Params: {trainable_params:,}")
        print(f"Trainable Ratio: {100 * trainable_params / all_params:.2f}%")

        # Check first layer type
        first_layer = self.vit.encoder.layers[0].mlp[0]
        print(f"Encoder Layer Type: {type(first_layer).__name__}")