"""GNN+ block 6: positional encoding, RWSE (random-walk structural encoding).

Injected as [x || x_RWSE] -> node_proj (Eq.12). RWSE is chosen over Laplacian
PE because Laplacian eigenvectors are sign/basis-ambiguous and change
completely when a subgraph is cut out, while RWSE is computable per subgraph
and stable."""

import torch


def random_walk_structural_encoding(g, ksteps: int) -> torch.Tensor:
    """RWSE for a batched DGL graph: per graph, diagonals of powers of
    M = A_hat for k = 1..ksteps.

    A_hat = D^-1 (A + I) (row-normalized, self-loops included; multi-edges
    accumulate).
    Returns (total_nodes, ksteps) float32 in batched-graph node order.

    Dense per-graph powers suit superpixel-scale graphs (up to ~1000 nodes
    each); do not enable pos_enc on larger graphs.
    """
    device = g.device
    eu, ev = g.edges()
    bn = g.batch_num_nodes().tolist()
    be = g.batch_num_edges().tolist()
    outs = []
    off_n = off_e = 0
    for n, m in zip(bn, be):
        u = (eu[off_e:off_e + m] - off_n).long()
        v = (ev[off_e:off_e + m] - off_n).long()
        a = torch.zeros(n, n, device=device)
        a.index_put_((u, v), torch.ones(m, device=device), accumulate=True)
        a = a + torch.eye(n, device=device)
        m_hat = a / a.sum(dim=1, keepdim=True).clamp(min=1.0)
        diag = []
        p = m_hat
        for _ in range(ksteps):
            diag.append(p.diagonal())
            p = p @ m_hat
        outs.append(torch.stack(diag, dim=-1))   # (n, ksteps)
        off_n += n
        off_e += m
    if not outs:
        return torch.zeros(0, ksteps, device=device)
    return torch.cat(outs, dim=0)
