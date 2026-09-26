"""Reveal actions: the core reveal_node(v) protocol action and teleport."""

import logging
import random
from typing import List, Optional, Tuple

from .graph_state import GraphState

logger = logging.getLogger(__name__)


def reveal_node(state: GraphState, v: int) -> List[Tuple[int, int]]:
    """Apply reveal_node(v), mutating the graph state in place.

    (1) v becomes revealed; (2) edges between v and revealed nodes become
    known; (3) v's incident edges/neighbors join the frontier.

    Returns the newly discovered frontier edges [(revealed, unrevealed), ...].
    """
    if v in state.V_revealed:
        raise ValueError(f"node {v} already revealed")

    edges_to_remove: List[Tuple[int, int]] = list(
        state._frontier_by_unrevealed.get(v, ())
    )

    for (ru, _rv) in edges_to_remove:
        if state.has_edge_feat:
            key = (min(ru, v), max(ru, v))
            if key not in state.revealed_edge_keys:
                state.revealed_edge_keys.add(key)
                eid = state._edge_key_to_id.get(key)
                if eid is not None:
                    state._revealed_edge_ids.append(eid)

    for edge in edges_to_remove:
        state.frontier_edges.discard(edge)

    state._frontier_by_unrevealed.pop(v, None)

    state.V_revealed.add(v)
    state.reveal_history.append(v)
    state._increment_budget()

    state._frontier_degree.pop(v, None)

    state._remove_frontier_node(v)
    state._add_to_revealed(v)

    new_frontier: List[Tuple[int, int]] = []
    neighbors = state._get_neighbors(v)

    for nbr in neighbors:
        if nbr == v:
            continue

        if nbr in state.V_revealed:
            if state.has_edge_feat:
                key = (min(v, nbr), max(v, nbr))
                if key not in state.revealed_edge_keys:
                    state.revealed_edge_keys.add(key)
                    eid = state._edge_key_to_id.get(key)
                    if eid is not None:
                        state._revealed_edge_ids.append(eid)
        else:
            edge = (v, nbr)
            is_new = edge not in state.frontier_edges
            state.frontier_edges.add(edge)
            state._frontier_degree[nbr] += 1
            is_new_node = nbr not in state._frontier_by_unrevealed
            state._frontier_by_unrevealed[nbr].add(edge)
            if is_new:
                new_frontier.append(edge)
            if is_new_node:
                state._add_frontier_node(nbr)

    logger.debug(
        "reveal_node(%d): new frontier=%d, |V_revealed|=%d, |frontier|=%d",
        v, len(new_frontier), len(state.V_revealed), len(state.frontier_edges),
    )

    return new_frontier


def teleport_if_stuck(state: GraphState, rng: random.Random = None) -> Optional[int]:
    """Pseudo-random teleport to an unrevealed node when the frontier is
    empty but budget remains.

    Happens when the seed falls in an isolated small component that has been
    fully revealed; with ratio < 1 an unrevealed node must exist.
    The teleport target counts as one reveal (costs 1 budget) and its
    neighbors become the new frontier.
    Returns the target node, or None when nothing can be revealed.

    rng: optional random.Random for reproducibility; defaults to global random.
    """
    if not state.is_frontier_empty:
        return None
    if state.get_budget_used() >= (state._budget_limit or state.n_nodes_total):
        return None
    unrevealed = [v for v in range(state.n_nodes_total) if v not in state.V_revealed]
    if not unrevealed:
        return None
    v = (rng or random).choice(unrevealed)
    reveal_node(state, v)
    return v
