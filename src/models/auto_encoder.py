
import torch
import torch.nn as nn
import torchvision
from torch.nn import init


# ---- Registry ---- meant for the different AE types
AE_REGISTRY = {}

def register_AE_type(name):
    """Decorator to register a AE class."""
    def decorator(cls):
        AE_REGISTRY[name] = cls
        return cls
    return decorator

def get_AE_type(class_name):
    """Get a AE class from the registry."""
    return AE_REGISTRY.get(class_name, None)

# function that calls from outside
def initialize_AE(global_args, input_dim):
    ae_type = global_args['ae_type']
    if ae_type not in AE_REGISTRY:
        raise ValueError(f"Unknown AE type '{ae_type}'. "
                         f"Available: {list(AE_REGISTRY.keys())}")

    if ae_type == 'identity':
        return AE_REGISTRY[ae_type]()
    elif input_dim == global_args['ae_latent_dim']:
        return AE_REGISTRY['identity']()
    elif ae_type == 'linear':
        return AE_REGISTRY[ae_type](input_dim=input_dim, latent_dim=global_args['ae_latent_dim'])



@register_AE_type('identity')
class IdentityAE(nn.Module):
    """Base class for all bottlenecks."""
    def __init__(self):
        super().__init__()
        self.encoder_net = nn.Identity()
        self.decoder_net = nn.Identity()
        self.is_trained = False
    def encode(self, x):
        return self.encoder_net(x)

    def decode(self, x):
        return self.decoder_net(x)

    def forward(self, x):        return self.decode(self.encode(x))


@register_AE_type('linear')
class LinearAE(IdentityAE):
    """Linear compression + expansion."""

    def __init__(self, input_dim, latent_dim, dropout=0.01):
        super().__init__()

        # --- Modifications Start Here ---
        # The encoder_net now holds the sequence of encoding layers
        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # The decoder_net now holds the sequence of decoding layers
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.LayerNorm(input_dim)
        )

        # Initialize weights (referencing the layers within Sequential)
        # Note: Accessing layers by index requires knowing the structure
        init.kaiming_uniform_(self.encoder_net[0].weight, nonlinearity='relu')
        init.xavier_uniform_(self.decoder_net[0].weight)
        init.zeros_(self.encoder_net[0].bias)
        init.zeros_(self.decoder_net[0].bias)

