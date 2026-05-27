"""
Base ICGALayer — all attention mechanics in one place.
Subclasses only override _apply_physics_bias() and add their own parameters.
"""

import torch
import torch.nn as nn
from ._physics import NODE_FEAT_DIM, EDGE_FEAT_DIM, compute_inertia_coupling


class ICGALayerBase(nn.Module):
    """
    Graph attention layer with a physics-bias hook.

        logit_ij = Q·K/√d  +  w_e·e_ij  +  _apply_physics_bias(...)

    Subclasses implement _apply_physics_bias to inject M̃-derived signals.
    At init=0 for all physics params, degrades to standard graph transformer.
    """

    def __init__(self, hidden: int, n_heads: int):
        super().__init__()
        assert hidden % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = hidden // n_heads

        self.W_q = nn.Linear(hidden, hidden, bias=False)
        self.W_k = nn.Linear(hidden, hidden, bias=False)
        self.W_v = nn.Linear(hidden, hidden, bias=False)
        self.W_o = nn.Linear(hidden, hidden)

        self.edge_bias = nn.Linear(EDGE_FEAT_DIM, n_heads, bias=False)

        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff    = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.ReLU(),
            nn.Linear(hidden * 2, hidden),
        )

    # ── Subclasses override this ──────────────────────────────────────────────

    def _apply_physics_bias(self, logits, M_edge, src_idx, dst_idx):
        """
        logits  : (B, E, H)
        M_edge  : (B, E)   — M̃[src, dst] per edge
        src_idx : (B, E)
        dst_idx : (B, E)
        Returns : (B, E, H)
        """
        return logits   # no-op — pure transformer

    # ── Shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _get_M_edge(M_tilde, src_idx, dst_idx):
        B, max_nodes, _ = M_tilde.shape
        src_flat = src_idx * max_nodes + dst_idx
        return M_tilde.view(B, max_nodes * max_nodes).gather(1, src_flat)   # (B, E)

    # ── Forward (identical for all variants) ─────────────────────────────────

    def forward(self, h, edge_features, edge_index, n_edges, M_tilde):
        B, max_nodes, hidden = h.shape
        max_edges = edge_index.shape[2]
        H, D      = self.n_heads, self.d_head
        device    = h.device

        src_idx = edge_index[:, 0, :]
        dst_idx = edge_index[:, 1, :]

        Q = self.W_q(h).view(B, max_nodes, H, D)
        K = self.W_k(h).view(B, max_nodes, H, D)
        V = self.W_v(h).view(B, max_nodes, H, D)

        src_e = src_idx.unsqueeze(-1).unsqueeze(-1).expand(B, max_edges, H, D)
        dst_e = dst_idx.unsqueeze(-1).unsqueeze(-1).expand(B, max_edges, H, D)
        Q_e   = Q.gather(1, dst_e)
        K_e   = K.gather(1, src_e)
        V_e   = V.gather(1, src_e)

        logits = (Q_e * K_e).sum(-1) * (D ** -0.5)
        logits = logits + self.edge_bias(edge_features.float())

        M_edge = self._get_M_edge(M_tilde, src_idx, dst_idx)
        logits = self._apply_physics_bias(logits, M_edge, src_idx, dst_idx)

        # Numerically stable scatter-softmax (log-sum-exp trick)
        edge_mask = (torch.arange(max_edges, device=device).unsqueeze(0)
                     < n_edges.squeeze(-1).unsqueeze(-1))
        dst_h = dst_idx.unsqueeze(-1).expand(B, max_edges, H)

        logits    = logits.masked_fill(~edge_mask.unsqueeze(-1), float("-inf"))
        node_max  = torch.full((B, max_nodes, H), float("-inf"), device=device)
        node_max.scatter_reduce_(1, dst_h, logits, reduce="amax", include_self=True)
        node_max  = node_max.clamp(min=-1e9)
        max_edge  = node_max.gather(1, dst_h)

        logits_exp = (logits - max_edge).exp() * edge_mask.unsqueeze(-1).float()
        denom      = torch.zeros(B, max_nodes, H, device=device)
        denom.scatter_add_(1, dst_h, logits_exp)
        alpha      = logits_exp / denom.gather(1, dst_h).clamp(min=1e-6)

        weighted = alpha.unsqueeze(-1) * V_e
        out      = torch.zeros(B, max_nodes, H, D, device=device)
        dst_hd   = dst_idx.unsqueeze(-1).unsqueeze(-1).expand(B, max_edges, H, D)
        out.scatter_add_(1, dst_hd, weighted)
        out = self.W_o(out.reshape(B, max_nodes, hidden))

        h = self.norm1(h + out)
        h = self.norm2(h + self.ff(h))
        return h


class CGATEncoderBase(nn.Module):
    """
    Shared encoder: node_embed → ICGA layers → masked mean pool.
    Pass icga_cls to swap in any ICGALayerBase subclass.
    """

    def __init__(self, hidden: int, n_icga_layers: int, n_heads: int,
                 icga_cls=None):
        super().__init__()
        if icga_cls is None:
            icga_cls = ICGALayerBase
        self.node_embed  = nn.Linear(NODE_FEAT_DIM, hidden)
        self.icga_layers = nn.ModuleList(
            [icga_cls(hidden, n_heads) for _ in range(n_icga_layers)]
        )

    def forward(self, obs: dict, M_tilde: torch.Tensor) -> torch.Tensor:
        node_features = obs["node_features"].float()
        edge_index    = obs["edge_index"].long()
        edge_features = obs["edge_features"].float()
        n_nodes       = obs["n_nodes"].long()
        n_edges       = obs["n_edges"].long()

        B, max_nodes, _ = node_features.shape
        h = self.node_embed(node_features)

        for layer in self.icga_layers:
            h = layer(h, edge_features, edge_index, n_edges, M_tilde)

        node_mask = (torch.arange(max_nodes, device=h.device).unsqueeze(0)
                     < n_nodes.squeeze(-1).unsqueeze(-1))
        h = h * node_mask.unsqueeze(-1).float()
        return h.sum(dim=1) / n_nodes.float()
