"""Shuffled-physics control: uses the right M_tilde values on the wrong edges."""

import torch
import torch.nn as nn
from models.base_ppo import BasePPOPolicy
from ._physics import compute_inertia_coupling
from ._icga_base import ICGALayerBase, HGFNEncoderBase


class ICGALayer(ICGALayerBase):

    def __init__(self, hidden: int, n_heads: int):
        super().__init__(hidden, n_heads)
        self.physics_beta = nn.Parameter(torch.zeros(1))

    def _apply_physics_bias(self, logits, M_edge, src_idx, dst_idx):
        shuffled = torch.flip(M_edge, dims=[1])
        return logits + torch.tanh(self.physics_beta) * shuffled.unsqueeze(-1)


class HGFNShuffledPPOPolicy(BasePPOPolicy):
    """Control variant: physics magnitude is present but edge assignment is wrong."""

    VARIANT = "shuffled"

    def __init__(self, hidden: int = 128, n_icga_layers: int = 2,
                 n_heads: int = 2, max_links: int = 4,
                 max_force: float = 20.0):
        super().__init__(hidden=hidden, max_force=max_force)
        self.encoder = HGFNEncoderBase(hidden, n_icga_layers, n_heads,
                                       icga_cls=ICGALayer)

    def encode(self, obs: dict):
        return self.encoder(obs, compute_inertia_coupling(obs))
