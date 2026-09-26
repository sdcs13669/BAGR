"""Evidence reader (BAGR): stateless frontier-selection path + recurrent
evidence-accumulation path with a history-guided attention readout.

Design invariants:
1. Single encoder - the only GNN is the shared frontier encoder; the embedding
   of the newly revealed node (delta_t) is taken from the same forward pass.
2. Selection is stateless and identical to the base scoring pipeline - the
   evidence stream never enters selection scores.
3. Recursion lives only in the prediction head - LSTM/attention gradients come
   exclusively from the supervised stepwise CE (L_ev), never from the policy
   gradient.

Per-step evidence stream (per state):
  delta_t = encoder embedding of the node revealed at the previous step
            (step 0: the seed), taken lazily from the current forward pass
  h_t     = LSTM(delta_t, h_{t-1})                      evidence integrator
  z_t     = AttnPool(query=h_t, K/V=revealed embeds)    history-guided readout
  logits  = MLP([z_t; h_t])                             stepwise prediction
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..core.base_reader import BaseReader
from ..config import ReaderConfig
from ..gnn.frontier.graph_builder import FrontierGraphBuilder
from ..gnn.frontier.encoder import FrontierNodesEncoder
from ..engine.graph_state import GraphState
from .reader_mlp import ReaderMLP
from .gsae import compute_structural_features
from .common import stable_probs as _stable_probs


class HistoryPool(nn.Module):
    """Single-head scaled dot-product attention pooling.

    query (d,) history state, K/V (n, d) revealed node embeddings -> (d,).
    Returns a zero vector when there is no revealed evidence (n=0).
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        if kv.shape[0] == 0:
            return torch.zeros(self.d, device=query.device, dtype=query.dtype)
        q = self.q_proj(query)
        K = self.k_proj(kv)
        V = self.v_proj(kv)
        attn = torch.softmax(K @ q / (self.d ** 0.5), dim=0)
        return attn @ V


class EvidenceReader(BaseReader, nn.Module):
    """Evidence reader: stateless selection + recurrent evidence prediction.

    Structure is the base selection pipeline (frontier builder -> frontier
    encoder -> scorer MLP over [embedding; remaining-budget fraction;
    structural features]) plus three evidence modules (evidence_lstm /
    evidence_pool / evidence_head, ~+0.2M params at width 128).
    """

    def __init__(
        self,
        edge_feat_dim: int = 7,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        stochastic: bool = False,
        temperature: float = 1.0,
        node_feat_dim: int = 0,
        n_classes: int = 0,
        config: Optional[ReaderConfig] = None,
    ):
        super().__init__()
        if config is None:
            config = ReaderConfig.from_legacy(
                kind='evidence', edge_feat_dim=edge_feat_dim,
                node_feat_dim=node_feat_dim, hidden_dim=hidden_dim,
                n_layers=n_layers, dropout=dropout,
                stochastic=stochastic, temperature=temperature,
                n_classes=n_classes)
        config.validate()
        self.config = config
        self.stochastic = config.stochastic
        self.temperature = config.temperature

        self.frontier_builder = FrontierGraphBuilder()
        self.frontier_encoder = FrontierNodesEncoder(
            edge_feat_dim=config.edge_feat_dim,
            hidden_dims=config.encoder_hidden_dims,
            dropouts=config.encoder_dropouts,
            node_feat_dim=config.node_feat_dim,
        )
        self.reader_mlp = ReaderMLP(
            input_dim=config.hidden_dim + 1 + 3,
            hidden_dims=config.scorer_hidden_dims,
            dropouts=config.scorer_dropouts,
        )

        d = self.config.hidden_dim
        # Ablation variants: dropped modules are not created so their state
        # dict keys disappear; evidence_head is always present.
        self.evidence_variant = self.config.evidence_variant
        if self.evidence_variant not in ('no_lstm', 'v1'):
            self.evidence_lstm = nn.LSTM(d, d)            # seq=1, batch_first=False
        if self.evidence_variant not in ('no_pool', 'v1'):
            self.evidence_pool = HistoryPool(d)
        self.evidence_head = nn.Sequential(
            nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, self.config.n_classes),
        )

    # ------------------------------------------------------------------
    # Selection path (stateless)
    # ------------------------------------------------------------------

    def forward(
        self, graph_state: GraphState, sample: bool = False
    ):
        """Differentiable scoring pipeline over a single state.

        sample=False: returns (scores, probs, frontier_nodes)
        sample=True:  returns (scores, probs, frontier_nodes, action_idx, log_prob)
        """
        subgraph = self.frontier_builder.build(graph_state)
        frontier_embeds, frontier_nodes = self.frontier_encoder(subgraph)

        if len(frontier_nodes) == 0:
            if sample:
                return torch.empty(0), torch.empty(0), [], None, None
            return torch.empty(0), torch.empty(0), []

        n_f = frontier_embeds.shape[0]
        remaining = graph_state.get_budget_ratio_remaining()
        remaining_tensor = torch.full(
            (n_f, 1), remaining,
            device=frontier_embeds.device, dtype=frontier_embeds.dtype,
        )
        struct_all = compute_structural_features(subgraph.g, subgraph.n_revealed)
        struct_frontier = struct_all[subgraph.n_revealed:].to(
            device=frontier_embeds.device, dtype=frontier_embeds.dtype,
        )
        combined = torch.cat(
            [frontier_embeds, remaining_tensor, struct_frontier], dim=-1,
        )
        scores = self.reader_mlp(combined)
        probs = _stable_probs(scores)

        if sample:
            if self.temperature != 1.0:
                probs = _stable_probs(scores, self.temperature)
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
            return scores, probs, frontier_nodes, action_idx, log_prob

        return scores, probs, frontier_nodes

    def _policy_scores(self, frontier_embeds, subgraph, graph_state):
        """Concatenate budget fraction + structural features, then score."""
        n_f = frontier_embeds.shape[0]

        remaining = graph_state.get_budget_ratio_remaining()
        remaining_tensor = torch.full(
            (n_f, 1), remaining,
            device=frontier_embeds.device, dtype=frontier_embeds.dtype,
        )

        struct_all = compute_structural_features(subgraph.g, subgraph.n_revealed)
        struct_frontier = struct_all[subgraph.n_revealed:].to(
            device=frontier_embeds.device, dtype=frontier_embeds.dtype,
        )

        combined = torch.cat(
            [frontier_embeds, remaining_tensor, struct_frontier], dim=-1,
        )
        scores = self.reader_mlp(combined)
        return scores

    def forward_states(
        self, states: List[GraphState], temperature: float = 1.0,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, List[int]]]:
        """Batch encode + score multiple states (lockstep loops).

        Returns one (scores, probs, frontier_nodes) tuple per state; states
        with an empty frontier get (empty, empty, []).
        """
        if not states:
            return []

        subgraphs = [self.frontier_builder.build(s) for s in states]
        batch_results = self.frontier_encoder.forward_batch(subgraphs)

        results: List[Tuple[torch.Tensor, torch.Tensor, List[int]]] = []
        for k, (frontier_embeds, frontier_nodes) in enumerate(batch_results):
            if not frontier_nodes:
                results.append((torch.empty(0), torch.empty(0), []))
                continue

            scores = self._policy_scores(frontier_embeds, subgraphs[k], states[k])
            probs = _stable_probs(scores, temperature)
            results.append((scores, probs, frontier_nodes))

        return results

    def select(self, graph_state: GraphState) -> Optional[int]:
        """Pick the next node to reveal (argmax, or sampled when stochastic)."""
        scores, probs, frontier_nodes = self.forward(graph_state)

        if len(frontier_nodes) == 0:
            return None

        if self.stochastic:
            if self.temperature != 1.0:
                probs = _stable_probs(scores, self.temperature)
            idx = torch.distributions.Categorical(probs).sample().item()
        else:
            idx = probs.argmax().item()
        return frontier_nodes[idx]

    # ------------------------------------------------------------------
    # Evidence stream
    # ------------------------------------------------------------------

    def init_evidence(self, n: int, device) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Initial zero (h, c) for n states; lockstep loops maintain them
        per state and rebuild at rollout boundaries."""
        return [(torch.zeros(1, self.config.hidden_dim, device=device),
                 torch.zeros(1, self.config.hidden_dim, device=device))
                for _ in range(n)]

    def forward_states_evidence(
        self, states, temperature: float,
        hc_list: List[Tuple[torch.Tensor, torch.Tensor]],
        last_revealed: List[int],
        return_states: bool = False,
    ):
        """Selection + evidence stream in one shared GNN forward.

        hc_list holds each state's current (h, c); last_revealed holds the
        node revealed at the previous step (step 0: the seed) whose embedding
        is read lazily from this forward pass (no second encoder).

        Returns:
            results   : List[(scores, probs, frontier_nodes)] - identical to
                        forward_states;
            new_hc    : List[(h, c)] updated integrator states;
            ev_logits : List[Tensor (n_classes,)] stepwise predictions;
            step_states (only if return_states=True): List[(z_t, h_t)].
        """
        if not states:
            if return_states:
                return [], [], [], []
            return [], [], []

        subgraphs = [self.frontier_builder.build(s) for s in states]
        all_embeds = self.frontier_encoder.forward_all_batch(subgraphs)

        d = self.config.hidden_dim
        deltas, hs, cs = [], [], []
        for k, sg in enumerate(subgraphs):
            h, c = hc_list[k]
            revealed_sorted = states[k].get_revealed_sorted()
            lr = last_revealed[k]
            if lr is not None and lr in revealed_sorted:
                pos = revealed_sorted.index(lr)
                deltas.append(all_embeds[k][pos])
            else:
                deltas.append(torch.zeros(d, device=h.device, dtype=h.dtype))
            hs.append(h.squeeze(0))
            cs.append(c.squeeze(0))

        # Integrator: full/no_pool use the LSTM; no_lstm/v1 bypass recursion
        # (h_t = delta_t, (h, c) returned unchanged).
        if self.evidence_variant in ('no_lstm', 'v1'):
            h_new = torch.stack(deltas)
            new_hc = list(hc_list)
        else:
            # One batched LSTM step: input (1, N, d), state (1, N, d)
            _, (h_seq, c_seq) = self.evidence_lstm(
                torch.stack(deltas).unsqueeze(0),
                (torch.stack(hs).unsqueeze(0), torch.stack(cs).unsqueeze(0)),
            )
            h_new = h_seq[0]
            c_new = c_seq[0]
            new_hc = [(h_new[i:i + 1], c_new[i:i + 1]) for i in range(len(states))]

        results: List[Tuple[torch.Tensor, torch.Tensor, List[int]]] = []
        ev_logits: List[torch.Tensor] = []
        step_states: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for k, sg in enumerate(subgraphs):
            h_all = all_embeds[k]
            n_rev = sg.n_revealed
            frontier_embeds = h_all[n_rev:]
            frontier_nodes = sg.frontier_original_ids

            if frontier_nodes:
                scores = self._policy_scores(frontier_embeds, sg, states[k])
                probs = _stable_probs(scores, temperature)
                results.append((scores, probs, frontier_nodes))
            else:
                results.append((torch.empty(0), torch.empty(0), []))

            h_t = h_new[k]
            # Readout: full/no_lstm use the history-guided attention; no_pool/v1
            # degrade to unguided mean pooling over revealed embeddings.
            if self.evidence_variant in ('no_pool', 'v1'):
                r_t = h_all[:n_rev]
                z_t = (r_t.mean(dim=0) if r_t.shape[0] > 0 else
                       torch.zeros(d, device=h_t.device, dtype=h_t.dtype))
            else:
                z_t = self.evidence_pool(h_t, h_all[:n_rev])
            ev_logits.append(self.evidence_head(torch.cat([z_t, h_t])))
            if return_states:
                step_states.append((z_t, h_t))

        if return_states:
            return results, new_hc, ev_logits, step_states
        return results, new_hc, ev_logits

    @property
    def name(self) -> str:
        return 'evidence'

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        self._training = mode
        return self

    def eval(self):
        nn.Module.eval(self)
        self._training = False
        return self
