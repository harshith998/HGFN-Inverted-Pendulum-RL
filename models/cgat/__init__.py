"""
CGAT model variants — all share the same graph obs space and param count (±2 scalars).

Variants
--------
  base        — scalar β·M̃ per layer  (current best)
  perhead     — per-head β·M̃  (one scalar per attention head)
  directional — β_fwd/β_bwd  (separate scales for root→leaf / leaf→root edges)
  gravity     — scalar β·M̃  +  gravity torque injected into node embeddings
  perc        — scalar β·M̃  +  PERC critic with w_H init=1
  no_physics  — matched graph transformer control, no M̃ bias
  shuffled    — control with M̃ values assigned to the wrong edges

Usage
-----
    from models.cgat import load_cgat_variant
    policy = load_cgat_variant("base", hidden=128, n_icga_layers=2, n_heads=2)
"""

from .cgat_base        import CGATBasePPOPolicy
from .cgat_perhead     import CGATPerHeadPPOPolicy
from .cgat_directional import CGATDirectionalPPOPolicy
from .cgat_gravity     import CGATGravityPPOPolicy
from .cgat_perc        import CGATPercPPOPolicy
from .cgat_no_physics  import CGATNoPhysicsPPOPolicy
from .cgat_shuffled    import CGATShuffledPPOPolicy

VARIANTS: dict = {
    "base":        CGATBasePPOPolicy,
    "perhead":     CGATPerHeadPPOPolicy,
    "directional": CGATDirectionalPPOPolicy,
    "gravity":     CGATGravityPPOPolicy,
    "perc":        CGATPercPPOPolicy,
    "no_physics":  CGATNoPhysicsPPOPolicy,
    "shuffled":    CGATShuffledPPOPolicy,
}


def load_cgat_variant(variant: str, **kwargs):
    """Instantiate a CGAT policy by variant name."""
    if variant not in VARIANTS:
        raise ValueError(
            f"Unknown CGAT variant '{variant}'. "
            f"Choose from: {list(VARIANTS)}"
        )
    return VARIANTS[variant](**kwargs)
