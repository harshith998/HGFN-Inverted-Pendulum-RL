"""
HGFN Directional β — separate β for root→leaf and leaf→root edges.

    logit_ij = Q·K/√d  +  w_e·e_ij  +  tanh(β_fwd) · M̃ᵢⱼ   (if src < dst)
    logit_ij = Q·K/√d  +  w_e·e_ij  +  tanh(β_bwd) · M̃ᵢⱼ   (if src > dst)

Force/control propagates root→leaf; inertia aggregates leaf→root.
These two directions may benefit from different coupling scales.
Two scalars instead of one — captures directionality without the full RIM.
"""

import torch
import torch.nn as nn
from models.base_ppo import BasePPOPolicy
from ._physics import compute_inertia_coupling
from ._icga_base import ICGALayerBase, HGFNEncoderBase


class ICGALayer(ICGALayerBase):

    def __init__(self, hidden: int, n_heads: int):
        super().__init__(hidden, n_heads)
        self.physics_beta_fwd = nn.Parameter(torch.zeros(1))   # root → leaf
        self.physics_beta_bwd = nn.Parameter(torch.zeros(1))   # leaf → root

    def _apply_physics_bias(self, logits, M_edge, src_idx, dst_idx):
        # src_idx < dst_idx → forward edge (root→leaf, lower index to higher)
        is_fwd      = (src_idx < dst_idx).float()               # (B, E)
        beta_fwd    = torch.tanh(self.physics_beta_fwd)
        beta_bwd    = torch.tanh(self.physics_beta_bwd)
        beta_edge   = is_fwd * beta_fwd + (1 - is_fwd) * beta_bwd  # (B, E)
        return logits + beta_edge.unsqueeze(-1) * M_edge.unsqueeze(-1)


class HGFNDirectionalPPOPolicy(BasePPOPolicy):
    """Transformer + directional β: separate physics scales for fwd/bwd edges."""

    VARIANT = "directional"

    def __init__(self, hidden: int = 128, n_icga_layers: int = 2,
                 n_heads: int = 2, max_links: int = 4, max_force: float = 20.0):
        super().__init__(hidden=hidden, max_force=max_force)
        self.encoder = HGFNEncoderBase(hidden, n_icga_layers, n_heads,
                                       icga_cls=ICGALayer)

    def encode(self, obs: dict) -> torch.Tensor:
        return self.encoder(obs, compute_inertia_coupling(obs))
