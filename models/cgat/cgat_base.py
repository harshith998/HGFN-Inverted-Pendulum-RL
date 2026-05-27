"""
CGAT Base — scalar β·M̃ attention bias (current best model).

    logit_ij = Q·K/√d  +  w_e·e_ij  +  tanh(β) · M̃ᵢⱼ

One learned scalar β per ICGA layer, shared across all heads.
β init=0 → degrades to standard transformer at start of training.
"""

import torch
import torch.nn as nn
from models.base_ppo import BasePPOPolicy
from ._physics import compute_inertia_coupling
from ._icga_base import ICGALayerBase, CGATEncoderBase


class ICGALayer(ICGALayerBase):

    def __init__(self, hidden: int, n_heads: int):
        super().__init__(hidden, n_heads)
        self.physics_beta = nn.Parameter(torch.zeros(1))

    def _apply_physics_bias(self, logits, M_edge, src_idx, dst_idx):
        # logits: (B, E, H),  M_edge: (B, E)
        return logits + torch.tanh(self.physics_beta) * M_edge.unsqueeze(-1)


class CGATBasePPOPolicy(BasePPOPolicy):
    """Transformer + single learned scalar β·M̃ per attention layer."""

    VARIANT = "base"

    def __init__(self, hidden: int = 128, n_icga_layers: int = 2,
                 n_heads: int = 2, max_links: int = 4, max_force: float = 20.0):
        super().__init__(hidden=hidden, max_force=max_force)
        self.encoder = CGATEncoderBase(hidden, n_icga_layers, n_heads,
                                       icga_cls=ICGALayer)

    def encode(self, obs: dict) -> torch.Tensor:
        return self.encoder(obs, compute_inertia_coupling(obs))
