"""No-physics control: same CGAT/ICGA attention stack, no M_tilde bias."""

from models.base_ppo import BasePPOPolicy
from ._physics import compute_inertia_coupling
from ._icga_base import ICGALayerBase, CGATEncoderBase


class CGATNoPhysicsPPOPolicy(BasePPOPolicy):
    """Graph transformer matched to CGAT depth/heads but with physics hook disabled."""

    VARIANT = "no_physics"

    def __init__(self, hidden: int = 128, n_icga_layers: int = 2,
                 n_heads: int = 2, max_links: int = 4,
                 max_force: float = 20.0):
        super().__init__(hidden=hidden, max_force=max_force)
        self.encoder = CGATEncoderBase(hidden, n_icga_layers, n_heads,
                                       icga_cls=ICGALayerBase)

    def encode(self, obs: dict):
        # M_tilde is computed to keep the same encoder call signature; ignored by layer.
        return self.encoder(obs, compute_inertia_coupling(obs))
