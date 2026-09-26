"""AgentNet baseline (AGR protocol adaptation, single agent).

A single-agent port of the official AgentNet implementation (ICLR 2023,
"Agent-based Graph Neural Networks", arXiv 2206.11010), constrained by the
AGR protocol to reveal one node per step. The movement head is Q.K attention
over [neighbor || current || edge] encodings plus a BSEU exploration bias
(back/stay/explored/unexplored learnable scalars, visited decay 0.9), with
gumbel-softmax (hard, straight-through) edge selection; node/agent residual
updates and a stepwise readout accumulated by averaging produce the
classification.

Protocol adaptation: (1) num_agents=1 (original K=18-26) so training and
evaluation are homogeneous and the budget semantics align exactly; (2) no
time conditioning (the original time-embedding table assumes a fixed step
count); (3) training on full graphs with end-to-end classification
supervision, while at evaluation the movement candidates are restricted to
frontier neighbors of the current position (depth-type, equivalent to the
random-walk candidate structure) with frontier node features masked and
one-ended edge features fed to the movement head; (4) dead ends navigate
within the revealed region by BSEU scores (no budget cost) until frontier
candidates exist; (5) deterministic evaluation via argmax with ties broken
toward the smaller id (no gumbel noise). own_predict is deterministic
(argmax moves, seeded start row); the batched forward (forward_batch)
produces identical logits for identical inputs."""

import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import dgl

_CODE_DIR = Path(__file__).resolve().parents[1] / 'code'
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from active_graph_reader.core.base_reader import BaseReader  # noqa: E402
from active_graph_reader.engine.graph_state import GraphState  # noqa: E402


def symmetrize_with_feat(g: dgl.DGLGraph) -> dgl.DGLGraph:
    """Symmetrize while keeping ndata/edata[feat] (dgl.to_bidirected drops
    edge features).

    Tensorized: dedup by the (min,max) canonical key (first edge wins),
    add the reverse direction for every undirected edge, and copy edge
    features to both directions. Output edges are in (min,max) lexicographic
    order; the model is permutation-invariant over edge order.
    """
    src, dst = g.edges()
    lo = torch.minimum(src, dst)
    hi = torch.maximum(src, dst)
    keys = torch.stack([lo, hi], dim=1)
    uniq, inv = torch.unique(keys, dim=0, return_inverse=True)
    first = torch.empty(uniq.shape[0], dtype=torch.long, device=keys.device)
    first.scatter_reduce_(0, inv, torch.arange(keys.shape[0], device=keys.device),
                          reduce='amin', include_self=False)
    u, v = uniq[:, 0], uniq[:, 1]
    g2 = dgl.graph((torch.cat([u, v]), torch.cat([v, u])),
                   num_nodes=g.num_nodes())
    if 'feat' in g.edata:
        s2, d2 = g2.edges()
        code_u = uniq[:, 0].long() * g.num_nodes() + uniq[:, 1].long()
        code_e = (torch.minimum(s2, d2) * g.num_nodes()
                  + torch.maximum(s2, d2))
        rows = torch.searchsorted(code_u, code_e)
        g2.edata['feat'] = g.edata['feat'][first[rows]]
    if 'feat' in g.ndata:
        g2.ndata['feat'] = g.ndata['feat']
    return g2


def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float) -> nn.Sequential:
    """Original conv_mlp/node_mlp/agent_mlp shape: pre-LN -> Linear -> act -> Linear."""
    layers = [nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.LeakyReLU(0.01)]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden, out_dim))
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class AgentNetModel(nn.Module):
    """Single-agent AgentNet (single-graph forward; self-loops required = stay)."""

    def __init__(self, node_feat_dim: int, edge_feat_dim: int, n_classes: int,
                 hidden: int = 128, num_steps: int = 16, dropout: float = 0.0,
                 visited_decay: float = 0.9, input_mlp: bool = True,
                 gumbel_tau: float = 2.0 / 3.0, mlp_width_mult: int = 1,
                 reduce: str = 'sum'):
        super().__init__()
        self.edge_feat_dim = edge_feat_dim
        self.hidden = hidden
        self.num_steps = num_steps
        self.dropout = dropout
        self.visited_decay = visited_decay
        self.gumbel_tau = gumbel_tau
        self.with_edge = edge_feat_dim > 0
        self.mlp_width_mult = int(mlp_width_mult)
        self.reduce = reduce
        assert self.reduce in ('sum', 'log'), f'unknown reduce: {reduce}'

        act = nn.LeakyReLU(0.01)
        w = self.mlp_width_mult
        if node_feat_dim > 0:
            if input_mlp:
                self.input_proj = nn.Sequential(
                    nn.Linear(node_feat_dim, hidden * 2), act,
                    nn.Linear(hidden * 2, hidden))
            else:
                self.input_proj = nn.Linear(node_feat_dim, hidden)
        else:
            self.input_proj = None
            self.node_token = nn.Parameter(torch.zeros(hidden))
        if self.with_edge:
            self.edge_input_proj = nn.Sequential(
                nn.Linear(edge_feat_dim, hidden * 2), act,
                nn.Linear(hidden * 2, hidden))

        self.agent_emb = nn.Embedding(1, hidden)
        key_in = hidden * 3 if self.with_edge else hidden * 2
        self.key = nn.Sequential(nn.LayerNorm(key_in), nn.Linear(key_in, hidden))
        self.query = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
        self.back_param = nn.Parameter(torch.tensor(0.0))
        self.stay_param = nn.Parameter(torch.tensor(-1.0))
        self.explored_param = nn.Parameter(torch.tensor(0.0))
        self.unexplored_param = nn.Parameter(torch.tensor(5.0))
        ed = hidden if self.with_edge else 0
        self.agent_node_lin = (nn.Sequential(
            nn.LayerNorm(hidden + ed), nn.Linear(hidden + ed, hidden), act)
            if self.with_edge else nn.Identity())
        self.node_mlp = _mlp(hidden * 2, hidden * 2 * w, hidden, dropout)
        self.message_val = (nn.Sequential(
            nn.LayerNorm(hidden + ed), nn.Linear(hidden + ed, hidden), act)
            if self.with_edge else None)
        self.conv_mlp = _mlp(hidden * 2, hidden * 2 * w, hidden, dropout)
        self.agent_mlp = _mlp(hidden * 2 + ed, hidden * 2 * w, hidden, dropout)
        self.step_readout = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden * 2),
            nn.Dropout(dropout))
        self.fc = nn.Linear(hidden * 2, n_classes)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def init_hidden(self, g: dgl.DGLGraph) -> torch.Tensor:
        if self.input_proj is not None:
            return self.input_proj(g.ndata['feat'])
        return self.node_token.unsqueeze(0).expand(g.num_nodes(), -1)

    def init_edge_emb(self, g: dgl.DGLGraph):
        if not self.with_edge:
            return None
        return self.edge_input_proj(g.edata['feat'])

    def new_agent(self, device) -> torch.Tensor:
        return self.agent_emb(torch.zeros(1, dtype=torch.long, device=device))[0]

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def move_logits(self, h: torch.Tensor, agent: torch.Tensor, cur: int,
                    cand: torch.Tensor, cand_edge: torch.Tensor = None,
                    cand_visit: torch.Tensor = None,
                    cand_is_prev: torch.Tensor = None) -> torch.Tensor:
        """Score candidate moves: attention + BSEU bias.

        stay_param applies when candidates include cur (self-loop stay);
        cand_is_prev marks the previous position (back_param)."""
        q = self.query(agent)
        cur_h = h[cur].unsqueeze(0).expand(len(cand), -1)
        k_in = torch.cat([h[cand], cur_h], dim=-1) if not self.with_edge \
            else torch.cat([h[cand], cand_edge, cur_h], dim=-1)
        k = self.key(k_in)
        attn = (q.unsqueeze(0) * k).sum(-1) / math.sqrt(self.hidden)
        bias = torch.zeros_like(attn)
        if cand_visit is not None:
            is_cur = cand == cur
            mask_old = ~is_cur
            s = cand_visit[mask_old]
            bias[mask_old] += (s / self.visited_decay) * self.explored_param \
                + (1.0 - s / self.visited_decay) * self.unexplored_param
            bias[is_cur] += self.stay_param
            if cand_is_prev is not None:
                bias[cand_is_prev] += self.back_param
        return attn + bias

    @torch.no_grad()
    def move_choice_eval(self, logits: torch.Tensor) -> int:
        """Evaluation move: argmax (deterministic, no gumbel noise)."""
        return int(logits.argmax().item())

    def move_choice_train(self, logits: torch.Tensor):
        """gumbel-softmax (hard, straight-through); returns (idx, attn_val)."""
        gumbels = -torch.log(torch.rand_like(logits).clamp_min(1e-10))
        y_soft = torch.softmax((logits + gumbels) / self.gumbel_tau, dim=-1)
        idx = int(y_soft.argmax().item())
        onehot = torch.zeros_like(y_soft)
        onehot[idx] = 1.0
        ret = onehot - y_soft.detach() + y_soft      # straight-through
        return idx, ret[idx]

    def _reduce(self, agg: torch.Tensor, count):
        """Aggregation reduce (original util.scatter semantics).

        'log' = sum/log2(count+1) (degree normalization; identity for
        count <= 1) where count is the number of messages aggregated into the
        node (convolution = neighbor count). 'sum' is the identity."""
        if self.reduce != 'log':
            return agg
        if isinstance(count, int):
            return agg / (math.log2(count + 1) if count > 0 else 1.0)
        cnt = count.clamp_min(1).float().unsqueeze(-1)
        return agg / torch.log2(cnt + 1.0)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def step_update(self, h: torch.Tensor, agent: torch.Tensor, pos: int,
                    conv_nbr: torch.Tensor, conv_edge: torch.Tensor = None,
                    moved_edge: torch.Tensor = None,
                    attn_val=1.0):
        """Return the updated (h, agent); functional update (index_copy,
        no in-place ops), safe across autograd steps."""
        if self.with_edge:
            me = moved_edge * attn_val
            a_msg = self.agent_node_lin(torch.cat([agent, me], dim=-1))
        else:
            a_msg = agent
        new_h_pos = h[pos] + self.node_mlp(torch.cat([h[pos], a_msg]))
        if conv_nbr.numel() > 0:
            if self.with_edge:
                agg = self.message_val(
                    torch.cat([h[conv_nbr], conv_edge], dim=-1)).sum(dim=0)
            else:
                agg = h[conv_nbr].sum(dim=0)
            agg = self._reduce(agg, int(conv_nbr.numel()))
        else:
            agg = torch.zeros(self.hidden, device=h.device)
        new_h_pos = new_h_pos + self.conv_mlp(torch.cat([new_h_pos, agg]))
        h = h.index_copy(0, torch.tensor([pos], dtype=torch.long,
                                          device=h.device),
                         new_h_pos.unsqueeze(0))
        if self.with_edge:
            agent_cat = torch.cat([agent, new_h_pos * attn_val, me])
        else:
            agent_cat = torch.cat([agent, new_h_pos * attn_val])
        return h, agent + self.agent_mlp(agent_cat)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def forward(self, g: dgl.DGLGraph, start: int = None,
                steps: int = None) -> torch.Tensor:
        h = self.init_hidden(g)
        edge_emb = self.init_edge_emb(g)
        n = g.num_nodes()
        device = h.device

        src, dst = g.edges()
        edge_map = {}
        nbrs = {}
        for e in range(g.num_edges()):
            u, v = int(src[e]), int(dst[e])
            edge_map.setdefault((u, v), e)
            nbrs.setdefault(u, set()).add(v)
            nbrs.setdefault(v, set()).add(u)

        agent = self.new_agent(device)
        cur = random.randrange(n) if start is None else int(start)
        visit = torch.zeros(n, device=device)
        visit[cur] = 1.0
        prev = None
        attn_val = 1.0
        moved_edge = torch.zeros(self.hidden, device=device) \
            if self.with_edge else None
        out = None

        T = self.num_steps if steps is None else int(steps)
        for i in range(T + 1):
            if i > 0:
                cand = sorted(nbrs.get(cur, set()))
                cand_t = torch.tensor(cand, dtype=torch.long, device=device)
                cand_e = None
                if self.with_edge:
                    cand_e = torch.stack([
                        edge_emb[edge_map[(cur, u)]] for u in cand])
                is_prev = (cand_t == prev) if prev is not None else None
                logits = self.move_logits(
                    h, agent, cur, cand_t, cand_e, visit[cand_t], is_prev)
                idx, attn_val = self.move_choice_train(logits)
                new_cur = cand[idx]
                if self.with_edge:
                    moved_edge = edge_emb[edge_map[(cur, new_cur)]]
                prev, cur = cur, new_cur
                visit = visit * self.visited_decay
                visit[cur] = 1.0
            conv_nbr = sorted(nbrs.get(cur, set()) - {cur})
            conv_nbr_t = torch.tensor(conv_nbr, dtype=torch.long,
                                      device=device)
            conv_e = None
            if self.with_edge:
                conv_e = torch.stack([
                    edge_emb[edge_map[(cur, u)]] for u in conv_nbr]) \
                    if conv_nbr else torch.zeros(0, self.hidden, device=device)
            h, agent = self.step_update(h, agent, cur, conv_nbr_t, conv_e,
                                        moved_edge, attn_val)
            layer_out = self.fc(self.step_readout(agent))
            out = layer_out if out is None else out + layer_out
        return (out / (T + 1)).unsqueeze(0)   # [1, n_classes]

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward_batch(self, g: dgl.DGLGraph, starts=None,
                      steps: int = None) -> torch.Tensor:
        """Deterministic evaluation readout on a block-diagonal batch graph
        (a dgl.batch product; self-loops required).

        Moves are argmax with ties broken toward the smaller id (no gumbel
        noise); the start is each graph's first node or a caller-provided row
        per graph (AGR passes the seed row within the revealed subgraph);
        attn_val is 1 (the straight-through limit of a hard one-hot). The math
        matches the single-graph forward exactly: padding is masked with -inf
        for argmax and zeroed for neighborhood aggregation. Identical inputs
        produce identical logits. Returns [B, n_classes]."""
        h = self.init_hidden(g)
        edge_emb = self.init_edge_emb(g)
        device = h.device
        B = g.batch_size

        src, dst = g.edges()
        nbr_eid = {}
        for e, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
            nbr_eid.setdefault(u, {}).setdefault(v, e)
            nbr_eid.setdefault(v, {}).setdefault(u, e)
        nbr_lists = {u: sorted(d.items()) for u, d in nbr_eid.items()}

        def _pad(seqs, width):
            """[(v, eid)] sequences -> (node ids [B,W], edge ids [B,W], valid mask [B,W])."""
            rows, pos, vals, eids = [], [], [], []
            for r, seq in enumerate(seqs):
                rows += [r] * len(seq)
                pos += list(range(len(seq)))
                vals += [v for v, _ in seq]
                eids += [e for _, e in seq]
            t_v = torch.zeros(len(seqs), width, dtype=torch.long, device=device)
            t_e = torch.zeros(len(seqs), width, dtype=torch.long, device=device)
            t_m = torch.zeros(len(seqs), width, dtype=torch.bool, device=device)
            if rows:
                r_t = torch.tensor(rows, device=device)
                p_t = torch.tensor(pos, device=device)
                t_v[r_t, p_t] = torch.tensor(vals, device=device)
                t_e[r_t, p_t] = torch.tensor(eids, device=device)
                t_m[r_t, p_t] = True
            return t_v, t_e, t_m

        agent = self.agent_emb(torch.zeros(B, dtype=torch.long, device=device))
        moved_edge = (torch.zeros(B, self.hidden, device=device)
                      if self.with_edge else None)
        visit = torch.zeros(g.num_nodes(), device=device)
        bnn = g.batch_num_nodes()
        cur_t = torch.zeros(B, dtype=torch.long, device=device)
        cur_t[1:] = torch.cumsum(bnn, 0)[:-1]
        if starts is not None:
            cur_t = cur_t + torch.as_tensor(starts, dtype=torch.long,
                                            device=device)
        visit[cur_t] = 1.0
        cur = cur_t.tolist()
        prev_t = None
        attn_val = torch.ones(B, device=device)
        out = None

        T = self.num_steps if steps is None else int(steps)
        for i in range(T + 1):
            if i > 0:
                seqs = [nbr_lists[c] for c in cur]
                D = max(len(s) for s in seqs)
                cand_t, cand_eid, valid = _pad(seqs, D)
                q = self.query(agent)
                cur_feat = h[cur_t]
                if self.with_edge:
                    k_in = torch.cat([h[cand_t], edge_emb[cand_eid],
                                      cur_feat.unsqueeze(1).expand(-1, D, -1)],
                                     dim=-1)
                else:
                    k_in = torch.cat([h[cand_t],
                                      cur_feat.unsqueeze(1).expand(-1, D, -1)],
                                     dim=-1)
                k = self.key(k_in)
                attn = (q.unsqueeze(1) * k).sum(-1) / math.sqrt(self.hidden)
                bias = torch.zeros_like(attn)
                is_cur = cand_t == cur_t.unsqueeze(1)
                s = visit[cand_t] / self.visited_decay
                term = s * self.explored_param \
                    + (1.0 - s) * self.unexplored_param
                bias = bias + term * (~is_cur & valid)
                bias = bias + self.stay_param * is_cur
                if prev_t is not None:
                    bias = bias + self.back_param * \
                        ((cand_t == prev_t.unsqueeze(1)) & valid)
                logits = (attn + bias).masked_fill(~valid, float('-inf'))
                idx = logits.argmax(dim=-1)
                attn_val = torch.ones(B, device=device)
                sel = idx.unsqueeze(1)
                prev_t = cur_t
                cur_t = cand_t.gather(1, sel).squeeze(1)
                cur = cur_t.tolist()
                if self.with_edge:
                    moved_edge = edge_emb[cand_eid.gather(1, sel).squeeze(1)]
                visit = visit * self.visited_decay
                visit[cur_t] = 1.0

            conv_seqs = [[(v, e) for v, e in nbr_lists[c] if v != c]
                         for c in cur]
            Dc = max(len(s) for s in conv_seqs)
            if Dc > 0:
                conv_t, conv_eid, conv_valid = _pad(conv_seqs, Dc)
                if self.with_edge:
                    agg = (self.message_val(
                        torch.cat([h[conv_t], edge_emb[conv_eid]], dim=-1))
                        * conv_valid.unsqueeze(-1)).sum(dim=1)
                else:
                    agg = (h[conv_t] * conv_valid.unsqueeze(-1)).sum(dim=1)
                agg = self._reduce(agg, conv_valid.sum(dim=1))
            else:
                agg = torch.zeros(B, self.hidden, device=device)
            if self.with_edge:
                me = moved_edge * attn_val.unsqueeze(-1)
                a_msg = self.agent_node_lin(torch.cat([agent, me], dim=-1))
            else:
                a_msg = agent
            cur_feat = h[cur_t]
            new_h_pos = cur_feat + self.node_mlp(
                torch.cat([cur_feat, a_msg], dim=-1))
            new_h_pos = new_h_pos + self.conv_mlp(
                torch.cat([new_h_pos, agg], dim=-1))
            h = h.index_copy(0, cur_t, new_h_pos)
            if self.with_edge:
                agent_cat = torch.cat(
                    [agent, new_h_pos * attn_val.unsqueeze(-1), me], dim=-1)
            else:
                agent_cat = torch.cat(
                    [agent, new_h_pos * attn_val.unsqueeze(-1)], dim=-1)
            agent = agent + self.agent_mlp(agent_cat)
            layer_out = self.fc(self.step_readout(agent))
            out = layer_out if out is None else out + layer_out
        return out / (T + 1)                  # [B, n_classes]


class _GraphCache:
    """Per-graph static cache shared across rollouts: full-graph encodings
    and the undirected adjacency.

    Equivalent to rebuilding the visible graph and re-encoding every step:
    node/edge features are static and the projections are row-wise
    independent, so one full-graph forward followed by row gathers matches
    per-step re-encoding; a masked frontier node encodes to the same constant
    row. Rebuilt when the graph changes (naturally hit in per-graph eval)."""

    __slots__ = ('adj', 'x_proj', 'e_proj', 'zero_row')

    def __init__(self, model: 'AgentNetModel', state: GraphState, device):
        g = state.g
        src, dst = g.edges()
        adj = {}
        for u, v in zip(src.tolist(), dst.tolist()):
            if u == v:
                continue
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        self.adj = {k: sorted(s) for k, s in adj.items()}
        if model.input_proj is not None:
            self.x_proj = model.input_proj(g.ndata['feat'].to(device))
            self.zero_row = model.input_proj(
                torch.zeros(1, state.node_feat_dim, device=device))
        else:
            self.x_proj = None
            self.zero_row = None
        self.e_proj = (model.edge_input_proj(g.edata['feat'].to(device))
                       if model.with_edge else None)

    def edge_id(self, state: GraphState, u: int, v: int) -> int:
        """Original edge id of (u,v); canonical min/max key as in GraphState."""
        return state._edge_key_to_id[(min(u, v), max(u, v))]


class AgentnetStrategy(BaseReader):
    """Single-agent walk revealing one node at a time (protocol-compliant
    evaluation form; stateful, rebuilt on reset).

    Fast path: per-graph static cache (_GraphCache) + row gathers instead of
    rebuilding a DGL graph and re-encoding every step; selection is bit-exact
    with the previous implementation."""

    NAV_CAP_FACTOR = 4

    def __init__(self):
        super().__init__()
        self._model = None
        self._device = None
        self._gc_graph = None
        self._gc = None          # _GraphCache
        self._reset_walk()

    def _reset_walk(self):
        self._pos = None
        self._prev = None
        self._visit = {}
        self._h_cache = {}
        self._agent = None
        self._nav_steps = 0

    @property
    def name(self) -> str:
        return 'agentnet'

    def load(self, ckpt_path: str, device='cpu') -> None:
        """Load a checkpoint trained by run_agentnet.py (structure from its config)."""
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        cfg = ckpt['config']
        model = AgentNetModel(
            node_feat_dim=cfg['node_feat_dim'],
            edge_feat_dim=cfg['edge_feat_dim'],
            n_classes=cfg['n_classes'],
            hidden=cfg['hidden'], num_steps=cfg['num_steps'],
            dropout=cfg['dropout'], visited_decay=cfg['visited_decay'],
            mlp_width_mult=cfg.get('mlp_width_mult', 1),
            reduce=cfg.get('reduce', 'sum'))
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        self._device = torch.device(device)
        self._model = model.to(self._device).eval()
        self._gc_graph = None
        self._gc = None

    def reset(self) -> None:
        self._reset_walk()

    def own_predict(self, sub_g, seed_row=None, steps=None):
        """Classify the revealed subgraph with the agent's own readout head.

        Recorded alongside the protocol judgment (the frozen classifier stays
        the terminal judge; this head is the reference, mirroring the
        acc_* / acc_clf_* convention). Input is the SubgraphBuilder product
        (symmetrized, self-loops, features); the walk is deterministic (argmax
        moves, seeded start row) -> [1, n_classes] logits, identical for
        identical subgraphs. seed_row is the row of the seed within the
        subgraph (index into builder original_ids), default 0."""
        return self.own_predict_batch([sub_g],
                                      [seed_row] if seed_row is not None else None,
                                      [steps] if steps is not None else None)

    def own_predict_batch(self, subgraphs, seed_rows=None, steps=None):
        """Batched own_predict: symmetrize -> dgl.batch -> lockstep walk.

        seed_rows: the seed row within each subgraph (None = first node);
        steps: readout walk length per subgraph (None = model num_steps).
        Unequal steps are grouped and forwarded separately (graphs are
        independent; results match per-graph runs). Returns [B, n_classes]."""
        gs = [dgl.add_self_loop(symmetrize_with_feat(dgl.remove_self_loop(s)))
              for s in subgraphs]
        B = len(gs)
        out = torch.zeros(B, self._model.fc.weight.shape[0],
                          device=self._device)
        if steps is None:
            groups = [(None, list(range(B)))]
        else:
            order = sorted(range(B), key=lambda i: steps[i])
            groups, i = [], 0
            while i < B:
                j = i
                while j < B and steps[order[j]] == steps[order[i]]:
                    j += 1
                groups.append((int(steps[order[i]]), order[i:j]))
                i = j
        for st_val, idx in groups:
            bg = dgl.batch([gs[k] for k in idx]).to(self._device)
            sr = ([seed_rows[k] for k in idx]
                  if seed_rows is not None else None)
            idx_t = torch.as_tensor(idx, dtype=torch.long,
                                    device=self._device)
            out[idx_t] = self._model.forward_batch(bg, starts=sr,
                                                   steps=st_val)
        return out

    def on_new_frontier(self, edges) -> None:
        """Infer the seed from initial_frontier and initialize the walk state."""
        if self._pos is None and edges:
            self._pos = edges[0][0]
            self._visit[self._pos] = 1.0
            self._agent = self._model.new_agent(self._device)

    def _edge_feat_of(self, state: GraphState, u: int, v: int):
        """Edge features of (u,v) by original id (frontier edges are visible)."""
        if not state.has_edge_feat:
            return None
        key = (min(u, v), max(u, v))
        return state.edge_feat[state._edge_key_to_id[key]]

    def _cand_from_frontier(self, state: GraphState):
        return sorted({v for (u, v) in state.frontier_edges if u == self._pos})

    @torch.no_grad()
    def select(self, graph_state: GraphState) -> 'int | None':
        frontier = graph_state.frontier_edges
        if not frontier:
            return None
        if self._model is None:
            raise RuntimeError("agentnet strategy needs load(checkpoint) "
                               "before use")
        if self._pos is None:
            self.on_new_frontier(sorted(frontier))

        m = self._model
        if self._gc_graph is not graph_state.g:
            self._gc_graph = graph_state.g
            self._gc = _GraphCache(m, graph_state, self._device)
        cache = self._gc

        revealed = graph_state.get_revealed_sorted()
        frontier_nodes = graph_state.get_frontier_nodes()
        ids = revealed + frontier_nodes
        row_of = {old: i for i, old in enumerate(ids)}
        if cache.x_proj is not None:
            h = torch.cat([
                cache.x_proj[revealed] if revealed else
                torch.zeros(0, m.hidden, device=self._device),
                cache.zero_row.expand(len(frontier_nodes), -1)])
        else:
            h = m.node_token.repeat(len(ids), 1)
        for old, hv in self._h_cache.items():
            h[row_of[old]] = hv

        cand = self._cand_from_frontier(graph_state)
        frontier_set = set(frontier_nodes)
        nav_cap = self.NAV_CAP_FACTOR * graph_state.n_nodes_total
        while not cand and self._nav_steps < nav_cap:
            self._nav_steps += 1
            nbrs = set(cache.adj.get(self._pos, ())) & graph_state.V_revealed
            if not nbrs:
                return None
            nbrs = sorted(nbrs)
            nbr_t = torch.tensor([row_of[v] for v in nbrs],
                                 dtype=torch.long, device=self._device)
            nbr_e = None
            if m.with_edge:
                nbr_e = torch.stack([
                    cache.e_proj[cache.edge_id(graph_state, self._pos, v)]
                    for v in nbrs])
            visit = torch.tensor([self._visit.get(v, 0.0) for v in nbrs],
                                 device=self._device)
            is_prev = torch.tensor([v == self._prev for v in nbrs],
                                   dtype=torch.bool, device=self._device) \
                if self._prev is not None else None
            logits = m.move_logits(
                h, self._agent, row_of[self._pos], nbr_t, nbr_e,
                visit, is_prev)
            nxt = nbrs[m.move_choice_eval(logits)]
            h = self._apply_move(graph_state, cache, row_of, h, nxt,
                                 frontier_set)
            cand = self._cand_from_frontier(graph_state)
        if not cand:
            return None

        cand_t = torch.tensor([row_of[v] for v in cand],
                              dtype=torch.long, device=self._device)
        cand_e = None
        if m.with_edge:
            cand_e = torch.stack([
                cache.e_proj[cache.edge_id(graph_state, self._pos, v)]
                for v in cand])
        visit = torch.tensor([self._visit.get(v, 0.0) for v in cand],
                             device=self._device)
        is_prev = torch.tensor([v == self._prev for v in cand],
                               dtype=torch.bool, device=self._device) \
            if self._prev is not None else None
        logits = m.move_logits(
            h, self._agent, row_of[self._pos], cand_t, cand_e,
            visit, is_prev)
        chosen = cand[m.move_choice_eval(logits)]
        h = self._apply_move(graph_state, cache, row_of, h, chosen,
                             frontier_set)
        return chosen

    def _apply_move(self, state: GraphState, cache: _GraphCache,
                    row_of: dict, h: torch.Tensor, target: int,
                    frontier_set: set) -> torch.Tensor:
        """Move to target (original id): agent/node updates + visit decay
        (the reveal itself happens in reveal_node). Returns the updated h
        (functional update).

        Convolution neighbors are the visible neighbors of target (original
        ids), ordered revealed (ascending) first then frontier (ascending) -
        identical to the previous implementation's aggregation order:
        - target revealed (navigation step): all its edges to revealed nodes
          and all real frontier edges are visible;
        - target unrevealed (reveal step): only frontier edges are visible,
          i.e. the revealed neighbors; frontier-frontier edges stay hidden and
          never enter the convolution.
        The legacy behavior on edge-feature-less graphs (empty
        revealed_edge_keys) is preserved as-is."""
        with torch.no_grad():
            m = self._model
            moved_edge = None
            if m.with_edge:
                moved_edge = cache.e_proj[
                    cache.edge_id(state, self._pos, target)]
            nbrs = cache.adj.get(target, ())
            if target in state.V_revealed:
                rev = ([v for v in nbrs if v in state.V_revealed]
                       if state.has_edge_feat else [])
                conv_orig = sorted(rev) + sorted(v for v in nbrs
                                                 if v in frontier_set)
            else:
                conv_orig = sorted(v for v in nbrs if v in state.V_revealed)
            conv_nbr = torch.tensor([row_of[v] for v in conv_orig],
                                    dtype=torch.long, device=self._device)
            conv_e = None
            if m.with_edge:
                conv_e = torch.stack([
                    cache.e_proj[cache.edge_id(state, target, v)]
                    for v in conv_orig]) \
                    if conv_orig else torch.zeros(
                        0, m.hidden, device=self._device)
            h, self._agent = m.step_update(
                h, self._agent, row_of[target], conv_nbr, conv_e, moved_edge)
            self._h_cache[target] = h[row_of[target]].clone()
            self._prev, self._pos = self._pos, target
            for k in self._visit:
                self._visit[k] *= m.visited_decay
            self._visit[target] = 1.0
        return h
