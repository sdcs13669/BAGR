"""TrajectoryGenerator: teachers pick top-n seeds, generate trajectories, and
filter by correct final classification.

Efficiency: one gradient pass per graph shared across gradient-family
teachers; generate_trajectory returns the final GraphState (no rebuild at
validation); final states are classified in one cross-graph mini-batch."""

import logging
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from active_graph_reader.engine.graph_state import GraphState
from ..teachers.gradient import GradientTeacher
from ..teachers.structural import StructuralTeacher
from .graph_meta import GraphMeta, precompute_graph_meta
from .trajectory import Teacher, Trajectory

logger = logging.getLogger(__name__)

_grad_teacher = GradientTeacher()
_struct_pagerank_teacher = StructuralTeacher(method='pagerank')


class TrajectoryGenerator:
    """Generate trajectories for every graph x teacher at a reveal ratio.

    Each teacher picks top-n_seeds seeds via score_nodes() and generates one
    trajectory per seed; validation runs as a cross-graph mini-batch.
    """

    def __init__(
        self,
        teachers: List[Teacher],
        classifier: torch.nn.Module,
        subgraph_builder,
        device: str = 'cpu',
        graph_batch_size: int = 32,
    ):
        self.teachers = teachers
        self.classifier = classifier
        self.subgraph_builder = subgraph_builder
        self.device = device
        self.graph_batch_size = graph_batch_size
        self.classifier.to(device)
        self.classifier.eval()

        teacher_names = [t.name for t in teachers]
        self._needs_gradient = any(
            n == 'gradient' or n.startswith('hybrid-') for n in teacher_names
        )
        self._needs_pagerank = 'structural-pagerank' in teacher_names
        self._hybrid_alphas = [
            float(n.split('-')[1].replace('a', ''))
            for n in teacher_names if n.startswith('hybrid-')
        ]

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def generate_all(
        self,
        dataset,
        train_idx: List[int],
        reveal_ratio: float,
        n_seeds: int = 5,
        ogb_to_local: Dict[int, int] = None,
    ) -> Dict[str, List[List[Trajectory]]]:
        """Generate all graph x teacher trajectories at a reveal ratio.


        train_idx: index list (possibly a shard slice). dataset is indexed in
        train_idx order (dataset[i] -> (g, label)); ogb_to_local maps
        local_pos <-> idx.
        """
        self.classifier.eval()

        if ogb_to_local is None:
            ogb_to_local = {ogb: pos for pos, ogb in enumerate(train_idx)}

        n_graphs = len(train_idx)
        logger.info("precomputing GraphMeta for %d graphs...", n_graphs)
        metas = [precompute_graph_meta(dataset[i][0])
                 for i in tqdm(range(n_graphs), desc="GraphMeta")]

        result: Dict[str, List[List[Trajectory]]] = {
            t.name: [[] for _ in range(n_graphs)] for t in self.teachers
        }

        total_pending = 0
        for batch_start in tqdm(range(0, n_graphs, self.graph_batch_size), desc="Batches"):
            batch_end = min(batch_start + self.graph_batch_size, n_graphs)
            # (teacher_name, result_g_idx, traj, state, label)
            pending: List[Tuple[str, int, Trajectory, GraphState, int]] = []

            for g_idx in range(batch_start, batch_end):
                ogb_idx = train_idx[g_idx]
                local_pos = ogb_to_local[ogb_idx]
                g, lbl = dataset[local_pos]
                label = int(lbl.item()) if hasattr(lbl, 'item') else int(lbl)
                meta = metas[g_idx]
                budget = max(1, int(reveal_ratio * meta.n_nodes))
                if budget < 1 or budget >= meta.n_nodes:
                    continue

                score_cache = self._compute_score_cache(g, label, meta)

                for teacher in self.teachers:
                    scores = score_cache.get(teacher.name)
                    trajs_states = self._generate_for_graph(
                        teacher, g, label, ogb_idx, meta, budget, n_seeds, scores=scores,
                    )
                    for traj, state in trajs_states:
                        pending.append((teacher.name, g_idx, traj, state, label))

            if pending:
                total_pending += len(pending)
                self._validate_batch(pending)

                for teacher_name, g_idx, traj, _state, _label in pending:
                    if traj.is_correct:
                        result[teacher_name][g_idx].append(traj)

        logger.info("Total trajectory candidates validated: %d", total_pending)
        return result

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def _compute_score_cache(
        self, graph, label: int, meta: GraphMeta,
    ) -> Dict[str, torch.Tensor]:
        """One gradient pass per graph, shared by gradient-family teachers."""
        cache: Dict[str, torch.Tensor] = {}

        if self._needs_gradient:
            grad_scores = _grad_teacher.score_nodes(graph, self.classifier, label)
            cache['gradient'] = grad_scores

            # Hybrid teachers: α·norm(grad) + (1-α)·norm(degree)
            deg = meta.degrees.float()
            max_deg = deg.max()
            deg_scores = deg / max_deg if max_deg > 0 else deg
            cache['structural-degree'] = deg_scores

            grad_norm = self._minmax_norm(grad_scores)
            for alpha in self._hybrid_alphas:
                key = f'hybrid-a{alpha:.2f}'
                cache[key] = alpha * grad_norm + (1.0 - alpha) * deg_scores
        else:
            deg = meta.degrees.float()
            max_deg = deg.max()
            cache['structural-degree'] = deg / max_deg if max_deg > 0 else deg

        if self._needs_pagerank:
            cache['structural-pagerank'] = _struct_pagerank_teacher.score_nodes(graph)

        return cache

    @staticmethod
    def _minmax_norm(t: torch.Tensor) -> torch.Tensor:
        t_min, t_max = t.min(), t.max()
        denom = t_max - t_min
        if denom > 0:
            return (t - t_min) / denom
        return torch.zeros_like(t)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def _generate_for_graph(
        self,
        teacher: Teacher,
        graph,
        label: int,
        idx: int,
        meta: GraphMeta,
        budget: int,
        n_seeds: int,
        scores: torch.Tensor = None,
    ) -> List[Tuple[Trajectory, GraphState]]:
        """Per teacher: pick top-n seeds via score_nodes, generate (traj, state).

        Passing scores skips teacher.score_nodes() (shared-score optimization).
        """
        if scores is None:
            scores = teacher.score_nodes(graph, self.classifier, label)

        top_seeds = scores.argsort(descending=True)[:n_seeds].tolist()

        results: List[Tuple[Trajectory, GraphState]] = []
        for seed_node in top_seeds:
            traj, state = teacher.generate_trajectory(
                graph, scores, seed_node, budget, edge_key_to_id=meta.edge_key_to_id,
            )
            traj.graph_idx = idx
            results.append((traj, state))

        return results

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def _validate_batch(
        self, pending: List[Tuple[str, int, Trajectory, GraphState, int]],
    ):
        """Classify all final subgraphs in the batch with one forward."""
        states = [item[3] for item in pending]
        labels = [item[4] for item in pending]

        batched_g, _ = self.subgraph_builder.build_batch(states)
        batched_g = batched_g.to(self.device)
        node_ids = torch.zeros(batched_g.num_nodes(), dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.classifier(batched_g, node_ids)

        label_t = torch.tensor(labels, device=self.device)
        for i, (teacher_name, g_idx, traj, _state, label) in enumerate(pending):
            logit_i = logits[i:i + 1]
            probs = F.softmax(logit_i, dim=-1)
            traj.final_pred = int(probs.argmax(dim=-1).item())
            traj.is_correct = (traj.final_pred == label)
            traj.ce = float(F.cross_entropy(logit_i, label_t[i:i + 1]))
