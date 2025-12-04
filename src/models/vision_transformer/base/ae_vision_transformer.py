import torch
from models.auto_encoder import IdentityAE
from models.meta_transformer.base.data2seq import InputModality
from models.vision_transformer.base.vision_transformer_base import VisionTransformerBase
from torch import nn

# Vision Transformer with Auto-Encoder Integration.
class AEVisionTransformer(nn.Module):
    def __init__(self, vit: VisionTransformerBase, auto_encoder: IdentityAE, split_layer: int, num_classes: int, device):
        super().__init__()

        self.vit = vit.vit
        self.split_layer = split_layer
        self.num_classes = num_classes
        self.device = device


        # 3. Insert AE into the encoder sequence
        encoder_layers = list(self.vit.encoder.layers)

        if split_layer < 0 or split_layer > len(encoder_layers):
            raise ValueError(f"Split layer must be between 0 and {len(encoder_layers)}")

        if auto_encoder is None:
            raise ValueError("Auto-encoder instance must be provided to AEVisionTransformer constructor.")

        encoder_layers.insert(split_layer, auto_encoder)
        self.vit.encoder.layers = nn.Sequential(*encoder_layers)

    def forward_full(self, x):
        x = x[InputModality.IMAGE]
        return self.vit(x)

    def retrieve_split_layer_activations(self, x):
        self.vit.eval()

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

        x = x + self.vit.pos_embedding
        x = self.vit.dropout(x)

        # 3. Forward Pass through Transformer Encoder up to split_layer
        for i in range(self.split_layer):
            x = self.vit.encoder.layers[i](x)


        return x

    # def embedding_layer(self, x):
    #
    #     n, c, h, w = x.shape
    #     p = self.vit.patch_size
    #     torch._assert(h == self.vit.image_size, f"Wrong image height! Expected {self.vit.image_size} but got {h}!")
    #     torch._assert(w == self.vit.image_size, f"Wrong image width! Expected {self.vit.image_size} but got {w}!")
    #     n_h = h // p
    #     n_w = w // p
    #
    #     # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
    #     x = self.vit.conv_proj(x)
    #     # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
    #     x = x.reshape(n, self.vit.hidden_dim, n_h * n_w)
    #
    #     # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
    #     # The self attention layer expects inputs in the format (N, S, E)
    #     # where S is the source sequence length, N is the batch size, E is the
    #     # embedding dimension
    #     x = x.permute(0, 2, 1)
    #
    #     return x

    # def forward_server(self, x):
    #     """
    #     Executes the model from the AE Decoder to the final classification head.
    #     Input x is the compressed latent representation.
    #     """
    #     x = x[InputModality.IMAGE]
    #
    #     # 1. AE Decoder
    #     ae_module = self.vit.encoder.layers[self.split_layer]
    #     x = ae_module.decode(x)
    #
    #     # 2. Remaining Transformer Blocks
    #     # Iterate from split_layer + 1 to the end
    #     total_layers = len(self.vit.encoder.layers)
    #     for i in range(self.split_layer + 1, total_layers):
    #         x = self.vit.encoder.layers[i](x)
    #
    #     # 3. Final Norm and Head
    #     x = self.vit.encoder.ln(x)
    #     x = self.vit.heads(x)
    #
    #     return x

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
