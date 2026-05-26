"""
HGFN model variants — all share the same graph obs space and param count (±2 scalars).

Variants
--------
  base        — scalar β·M̃ per layer  (current best)
  perhead     — per-head β·M̃  (H scalars per layer)
  directional — β_fwd/β_bwd  (separate scales for root→leaf / leaf→root edges)
  gravity     — scalar β·M̃  +  gravity torque injected into node embeddings
  perc        — scalar β·M̃  +  PERC critic with w_H init=1
  no_physics  — matched graph transformer control, no M̃ bias
  shuffled    — control with M̃ values assigned to the wrong edges

Usage
-----
    from models.hgfn import load_hgfn_variant
    policy = load_hgfn_variant("base", hidden=128, n_icga_layers=2, n_heads=2)
"""

from .hgfn_base        import HGFNBasePPOPolicy
from .hgfn_perhead     import HGFNPerHeadPPOPolicy
from .hgfn_directional import HGFNDirectionalPPOPolicy
from .hgfn_gravity     import HGFNGravityPPOPolicy
from .hgfn_perc        import HGFNPercPPOPolicy
from .hgfn_no_physics  import HGFNNoPhysicsPPOPolicy
from .hgfn_shuffled    import HGFNShuffledPPOPolicy

VARIANTS: dict = {
    "base":        HGFNBasePPOPolicy,
    "perhead":     HGFNPerHeadPPOPolicy,
    "directional": HGFNDirectionalPPOPolicy,
    "gravity":     HGFNGravityPPOPolicy,
    "perc":        HGFNPercPPOPolicy,
    "no_physics":  HGFNNoPhysicsPPOPolicy,
    "shuffled":    HGFNShuffledPPOPolicy,
}


def load_hgfn_variant(variant: str, **kwargs):
    """Instantiate an HGFN policy by variant name."""
    if variant not in VARIANTS:
        raise ValueError(
            f"Unknown HGFN variant '{variant}'. "
            f"Choose from: {list(VARIANTS)}"
        )
    return VARIANTS[variant](**kwargs)
