"""Soft-Mask GNN baseline (AGR protocol adaptation).

Reproduced from the official Soft-Mask implementation (WWW 2021, "Soft-Mask:
Adaptive Substructure Extractions for GNNs", PyG version), SMG variant: 3 x
(WeightConv1 mask generation -> SparseConv mask-modulated convolution),
add-pool readout, 2-layer MLP head; hyperparameters follow the original
defaults (hidden 128 / dropout 0.5 / lr 5e-4).

Protocol adaptation: (1) mask-generation neighborhood aggregation runs on the
full graph at training; at evaluation only the first-layer mask scores are
recomputed on the visible subgraph (V_revealed + frontier) and the
highest-scoring frontier node is revealed (ties to the smaller id); (2) the
original pipeline discards edge features, so a small MLP encodes each edge's
features into a scalar weight passed through the SparseConv edge_weight
channel (message = w_e * h_j); (3) aggregation excludes self-loops via a
not_self edge weight; frontier node features are masked; (4) node-feature-less
datasets use a shared token (matching sagpool.py). Training runs on full
graphs; the scorer only sees the visible subgraph at evaluation - the
distribution shift is part of the baseline setting."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

_CODE_DIR = Path(__file__).resolve().parents[1] / 'code'
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from active_graph_reader.core.base_reader import BaseReader  # noqa: E402
from active_graph_reader.engine.graph_state import GraphState  # noqa: E402


def _not_self_weight(g: dgl.DGLGraph, feat_weight: torch.Tensor = None) -> torch.Tensor:
    """Per-edge scalar message weight = edge-feature encoding x not-self indicator."""
    src, dst = g.edges()
    w = (src != dst).float()
    if feat_weight is not None:
        w = w * feat_weight
    return w


class MaskGen(nn.Module):
    """DGL port of WeightConv1: neighborhood aggregation + self features -> mask logit."""

    def __init__(self, in_dim: int, hid_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hid_dim)
        self.lin2 = nn.Linear(in_dim, hid_dim)
        self.lin3 = nn.Linear(hid_dim * 2, hid_dim)
        self.lin4 = nn.Linear(hid_dim, 1)

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        with g.local_scope():
            xs = x if mask is None else x * mask.unsqueeze(-1)
            g.ndata['_h'] = self.lin1(xs)
            g.update_all(fn.copy_u('_h', '_m'), fn.sum('_m', '_agg'))
            agg = g.ndata['_agg']
            w = self.lin4(F.relu(self.lin3(torch.cat([agg, self.lin2(xs)], dim=-1))))
            return w.squeeze(-1)          # pre-sigmoid logit


class MaskedConv(nn.Module):
    """DGL port of SparseConv: (x*m)W messages (edge-weight modulated) + lin(x) residual, output * m."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False)
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor, mask: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        with g.local_scope():
            g.ndata['_x'] = x
            g.ndata['_h'] = self.weight(x * mask.unsqueeze(-1))
            g.edata['_w'] = edge_weight
            g.update_all(fn.u_mul_e('_h', '_w', '_m'), fn.sum('_m', '_agg'))
            out = g.ndata['_agg'] + self.lin(x)
        return out * mask.unsqueeze(-1)


class SoftMaskModel(nn.Module):
    """Hierarchical soft mask (single-graph forward; self-loops required)."""

    def __init__(self, node_feat_dim: int, edge_feat_dim: int, n_classes: int,
                 hidden: int = 128, n_layers: int = 3, dropout: float = 0.5):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden = hidden
        self.n_layers = n_layers
        self.dropout = dropout

        if node_feat_dim > 0:
            self.node_enc = nn.Linear(node_feat_dim, hidden)
        else:
            self.node_enc = None
            self.node_token = nn.Parameter(torch.zeros(hidden))
        if edge_feat_dim > 0:
            self.edge_enc = nn.Sequential(
                nn.Linear(edge_feat_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, 1))
            nn.init.zeros_(self.edge_enc[-1].weight)
            nn.init.zeros_(self.edge_enc[-1].bias)
        else:
            self.edge_enc = None
        self.convs = nn.ModuleList(MaskedConv(hidden, hidden)
                                   for _ in range(n_layers))
        self.masks = nn.ModuleList(MaskGen(hidden, hidden)
                                   for _ in range(n_layers))
        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, n_classes)

    def init_nodes(self, g: dgl.DGLGraph) -> torch.Tensor:
        if self.node_enc is not None:
            return self.node_enc(g.ndata['feat'])
        return self.node_token.unsqueeze(0).expand(g.num_nodes(), -1)

    def edge_weights(self, g: dgl.DGLGraph) -> torch.Tensor:
        feat_w = None
        if self.edge_enc is not None and 'feat' in g.edata:
            feat_w = self.edge_enc(g.edata['feat']).squeeze(-1)
        return _not_self_weight(g, feat_w)

    def first_stage_scores(self, g: dgl.DGLGraph) -> torch.Tensor:
        """Protocol evaluation entry: first-layer mask per-node logits."""
        h = self.init_nodes(g)
        return self.masks[0](g, h, mask=None)

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        """Full SMG forward (single graph; chained mask-modulated layers)."""
        h = self.init_nodes(g)
        e = self.edge_weights(g)
        mask_val = None
        for i in range(self.n_layers):
            mask_logit = self.masks[i](g, h, mask_val)
            mask_val = torch.sigmoid(mask_logit)
            h = F.relu(self.convs[i](g, h, mask_val, e))
        with g.local_scope():
            g.ndata['_p'] = h
            out = dgl.sum_nodes(g, '_p')
        x = F.relu(self.lin1(out))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)                        # [1, n_classes]


class SoftmaskStrategy(BaseReader):
    """First-layer mask scores -> greedy frontier selection (protocol-compliant)."""

    def __init__(self):
        super().__init__()
        self._model = None
        self._device = None

    @property
    def name(self) -> str:
        return 'softmask'

    def load(self, ckpt_path: str, device='cpu') -> None:
        """Load a scorer checkpoint trained by run_softmask.py (structure from its config)."""
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        cfg = ckpt['config']
        model = SoftMaskModel(
            node_feat_dim=cfg['node_feat_dim'],
            edge_feat_dim=cfg['edge_feat_dim'],
            n_classes=cfg['n_classes'],
            hidden=cfg['hidden'], n_layers=cfg['n_layers'],
            dropout=cfg['dropout'])
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        self._device = torch.device(device)
        self._model = model.to(self._device).eval()

    def reset(self) -> None:
        pass

    def select(self, graph_state: GraphState) -> 'int | None':
        frontier = graph_state.frontier_edges
        if not frontier:
            return None
        if self._model is None:
            raise RuntimeError("softmask strategy needs load(checkpoint) "
                               "before use")
        frontier_nodes = sorted({v for (_u, v) in frontier})
        from baseline.sagpool import build_visible_graph
        g, ids = build_visible_graph(graph_state)
        with torch.no_grad():
            scores = self._model.first_stage_scores(g.to(self._device))
        score_of = {old: float(s) for old, s in zip(ids, scores.tolist())}
        return max(frontier_nodes, key=lambda v: (score_of[v], -v))

    @torch.no_grad()
    def own_predict_batch(self, subgraphs, seed_rows, steps):
        """Own-head judgment: SMG classifier on each final revealed subgraph
        (chained mask-modulated layers). Feeds the acc_own_* reference or the
        judge='own' terminal judgment; seed_rows/steps are unused placeholders
        for the batched interface."""
        outs = []
        for g in subgraphs:
            g = dgl.add_self_loop(dgl.remove_self_loop(g))
            if self._model.edge_feat_dim > 0 and 'feat' not in g.edata:
                g.edata['feat'] = torch.zeros(g.num_edges(),
                                              self._model.edge_feat_dim)
            if self._model.node_feat_dim > 0 and 'feat' not in g.ndata:
                g.ndata['feat'] = torch.zeros(g.num_nodes(),
                                              self._model.node_feat_dim)
            outs.append(self._model(g.to(self._device)))
        return torch.cat(outs, dim=0)
