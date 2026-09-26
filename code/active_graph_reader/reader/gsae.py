"""Structural features for subgraph nodes: 3 dims (normalized revealed degree,
signed normalized frontier degree, local clustering coefficient)."""

import torch


def compute_structural_features(g, n_revealed: int):
    """Compute the 3-dim structural features of every subgraph node.

    Features:
      dim 0: normalized revealed degree (connections to the revealed set)
      dim 1: signed normalized frontier degree (out_deg - in_deg) / max_abs
        in [-1, 1]
        positive for revealed nodes linking to the frontier
      dim 2: local clustering coefficient among revealed neighbors

    Returns (n_total, 3), normalized within the subgraph.
    """
    n_total = g.num_nodes()
    device = g.device
    in_deg = g.in_degrees().float()
    out_deg = g.out_degrees().float()

    deg_revealed = in_deg.clone()
    deg_frontier = out_deg - in_deg

    norm_deg = deg_revealed / deg_revealed.max().clamp(min=1)
    norm_fr_deg = deg_frontier / deg_frontier.abs().max().clamp(min=1)

    clustering = torch.zeros(n_total, device=device)
    if n_total > 0 and n_revealed > 0:
        adj = g.adjacency_matrix().to_dense()
        for v in range(n_revealed):
            d = int(deg_revealed[v].item())
            if d < 2:
                continue
            nbr_mask = (adj[v] > 0) & (torch.arange(n_total, device=device) < n_revealed)
            nbr = nbr_mask.nonzero(as_tuple=True)[0]
            nbr_count = nbr.numel()
            if nbr_count < 2:
                continue
            e_v = adj[nbr][:, nbr].sum().item() / 2.0
            clustering[v] = 2.0 * e_v / (nbr_count * (nbr_count - 1))

    return torch.stack([norm_deg, norm_fr_deg, clustering], dim=-1)
