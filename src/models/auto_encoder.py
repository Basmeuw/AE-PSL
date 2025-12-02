import torch
import torch.nn as nn
import torchvision
# from abc import ABC, abstractmethod
#
# # ---- Abstract Base class ----
# class AE(nn.Module, ABC):
#     """Base class for all bottlenecks."""
#     def encode(self, x):
#         return x
#
#     def decode(self, x):
#         return x
#
#     def forward(self, x):
#         return self.decode(self.encode(x))
#
# class IdentityAE(AE):
#     """Pass-through auto-encoder (no compression)."""
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         encoder_net = nn.Identity()
#         decoder_net = nn.Identity()
#
# ---- Base class ----
class IdentityAE(nn.Module):
    """Base class for all bottlenecks."""
    def __init__(self):
        super().__init__()
        self.encoder_net = nn.Identity()
        self.decoder_net = nn.Identity()

    def encode(self, x):
        return x

    def decode(self, x):
        return x

    def forward(self, x):
        return self.decode(self.encode(x))


