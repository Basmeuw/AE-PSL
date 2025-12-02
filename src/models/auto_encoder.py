import torch
import torch.nn as nn
import torchvision

# ---- Base class ----
class AE(nn.Module):
    """Base class for all bottlenecks."""
    def encoder(self, x):
        return x

    def decoder(self, x):
        return x

    def forward(self, x):
        return self.decoder(self.encoder(x))

