"""SAGPool baseline (AGR protocol adaptation).

Reproduced from the official SAGPool implementation (ICML 2019,
"Self-Attention Graph Pooling", PyG version) with the same skeleton: 3 x
(conv -> gate -> top-k gated pooling), per-layer mean+max readout summed
across layers, and a 3-layer MLP head; hyperparameters follow the original
defaults (hidden 128 / pool_ratio 0.5 / dropout 0.5).

Protocol adaptation: (1) the conv is a GINE-style edge-feature convolution
(edge features are the protocol-visible frontier information and PPA/COLLAB
have no node features); (2) the original one-shot top-k over the complete
graph violates the protocol, so at each step the first-stage gate scores are
recomputed on the visible subgraph (V_revealed + frontier) and the
highest-scoring frontier node is revealed (ties to the smaller id,
deterministic); (3) frontier node features are masked (identity and edge
features are protocol-visible, node features are not); (4) training runs on
full graphs (standard SAGPool supervision) while the scorer only sees the
visible subgraph at evaluation - the distribution shift is part of the
baseline setting."""

import math
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


class EGINConv(nn.Module):
    """GINE-style edge-feature conv: h_v' = MLP((1+eps)*h_v + sum relu(h_u + e_uv))."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU())

    def forward(self, g, h, e):
        with g.local_scope():
            src, _dst = g.edges()
            g.edata['m'] = F.relu(h[src] + e)
            g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'agg'))
            agg = g.ndata['agg']
        return self.mlp((1.0 + self.eps) * h + agg)


class GateConv(nn.Module):
    """GINE version of the SAGPool score_layer; outputs a per-node scalar."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.lin = nn.Linear(in_dim, 1)

    def forward(self, g, h, e):
        with g.local_scope():
            src, _dst = g.edges()
            g.edata['m'] = F.relu(h[src] + e)
            g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'agg'))
            agg = g.ndata['agg']
        return self.lin((1.0 + self.eps) * h + agg).squeeze(-1)


class SAGPoolModel(nn.Module):
    """Hierarchical SAGPool (single-graph forward; self-loops required)."""

    def __init__(self, node_feat_dim: int, edge_feat_dim: int, n_classes: int,
                 hidden: int = 128, n_layers: int = 3, pool_ratio: float = 0.5,
                 dropout: float = 0.5):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden = hidden
        self.n_layers = n_layers
        self.pool_ratio = pool_ratio
        self.dropout = dropout

        if node_feat_dim > 0:
            self.node_enc = nn.Linear(node_feat_dim, hidden)
        else:
            self.node_enc = None
            self.node_token = nn.Parameter(torch.zeros(hidden))
        self.edge_proj = (nn.Linear(edge_feat_dim, hidden)
                          if edge_feat_dim > 0 else None)
        self.convs = nn.ModuleList(EGINConv(hidden, hidden)
                                   for _ in range(n_layers))
        self.gates = nn.ModuleList(GateConv(hidden) for _ in range(n_layers))
        self.lin1 = nn.Linear(2 * hidden, hidden)
        self.lin2 = nn.Linear(hidden, hidden // 2)
        self.lin3 = nn.Linear(hidden // 2, n_classes)

    def init_nodes(self, g):
        if self.node_enc is not None:
            return self.node_enc(g.ndata['feat'])
        return self.node_token.unsqueeze(0).expand(g.num_nodes(), -1)

    def init_edges(self, g):
        if self.edge_proj is not None:
            return F.relu(self.edge_proj(g.edata['feat']))
        return g.edata['feat'].new_zeros(g.num_edges(), self.hidden)

    def first_stage_scores(self, g):
        """Protocol evaluation entry: first-stage gate per-node scores."""
        h = self.init_nodes(g)
        e = self.init_edges(g)
        h = F.relu(self.convs[0](g, h, e))
        return self.gates[0](g, h, e)

    def forward(self, g):
        """Full SAGPool forward (single graph; hierarchical top-k + summed readouts)."""
        h = self.init_nodes(g)
        e = self.init_edges(g)
        out = 0
        for i, (conv, gate) in enumerate(zip(self.convs, self.gates)):
            h = F.relu(conv(g, h, e))
            s = gate(g, h, e)
            k = max(1, math.ceil(self.pool_ratio * g.num_nodes()))
            idx = torch.topk(s, k, largest=True).indices
            h = h[idx] * torch.tanh(s[idx]).unsqueeze(-1)
            out = out + torch.cat([h.max(dim=0).values, h.mean(dim=0)])
            if i < self.n_layers - 1:
                g = dgl.node_subgraph(g, idx)
                e = e[g.edata[dgl.EID]]
        x = F.relu(self.lin1(out))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin2(x))
        return self.lin3(x).unsqueeze(0)


def build_visible_graph(state: GraphState):
    """Visible subgraph: V_revealed + frontier nodes, revealed edges (both
    directions) and frontier edges (both directions).

    Frontier node features are masked (identity and edge features are
    protocol-visible, node features are not); self-loops are normalized
    (remove+add, matching the training graphs).

    g: dgl.DGLGraph with ndata[feat]/edata[feat] when the original graph has
    them. ids: subgraph node i -> original node id (revealed ascending first,
    frontier ascending after)."""
    revealed = sorted(state.V_revealed)
    frontier_nodes = sorted({v for (_u, v) in state.frontier_edges})
    ids = revealed + frontier_nodes
    n = len(ids)
    old_to_new = {old: i for i, old in enumerate(ids)}

    src, dst, ekeys = [], [], []
    for u, v in list(state.revealed_edge_keys) + list(state.frontier_edges):
        key_u, key_v = min(u, v), max(u, v)
        nu, nv = old_to_new[key_u], old_to_new[key_v]
        src += [nu, nv]
        dst += [nv, nu]
        ekeys += [(key_u, key_v), (key_u, key_v)]

    g = dgl.graph((torch.tensor(src, dtype=torch.long),
                   torch.tensor(dst, dtype=torch.long)), num_nodes=n)
    if state.has_edge_feat:
        eids = [state._edge_key_to_id[k] for k in ekeys]
        g.edata['feat'] = state.edge_feat[eids]
    if state.has_node_feat:
        x = torch.zeros(n, state.node_feat_dim)
        x[:len(revealed)] = state.node_feat[revealed]
        g.ndata['feat'] = x
    g = dgl.add_self_loop(dgl.remove_self_loop(g))
    return g, ids


class SagpoolStrategy(BaseReader):
    """First-stage gate scores -> greedy frontier selection (protocol-compliant)."""

    def __init__(self):
        super().__init__()
        self._model = None
        self._device = None

    @property
    def name(self) -> str:
        return 'sagpool'

    def load(self, ckpt_path: str, device='cpu') -> None:
        """Load a scorer checkpoint trained by run_sagpool.py (structure from its config)."""
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        cfg = ckpt['config']
        model = SAGPoolModel(
            node_feat_dim=cfg['node_feat_dim'],
            edge_feat_dim=cfg['edge_feat_dim'],
            n_classes=cfg['n_classes'],
            hidden=cfg['hidden'], n_layers=cfg['n_layers'],
            pool_ratio=cfg['pool_ratio'], dropout=cfg['dropout'])
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
            raise RuntimeError("sagpool strategy needs load(checkpoint) "
                               "before use")
        frontier_nodes = sorted({v for (_u, v) in frontier})
        g, ids = build_visible_graph(graph_state)
        with torch.no_grad():
            scores = self._model.first_stage_scores(g.to(self._device))
        score_of = {old: float(s) for old, s in zip(ids, scores.tolist())}
        return max(frontier_nodes, key=lambda v: (score_of[v], -v))

    @torch.no_grad()
    def own_predict_batch(self, subgraphs, seed_rows, steps):
        """Own-head judgment: SAGPool classifier on each final revealed subgraph
        (single-graph BN convention). Feeds the acc_own_* reference or the
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
