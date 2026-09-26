"""Reveal protocol entry points: run_reveal / reveal_one / reveal_batch."""

import random
from typing import List, Optional, Tuple

import dgl

from ..core.base_reader import BaseReader
from .graph_state import GraphState
from .reveal_loop import RevealLoop


def run_reveal(
    reader: BaseReader,
    full_graph: dgl.DGLGraph,
    budget: int,
    seed_node: Optional[int] = None,
) -> GraphState:
    """Run the reveal protocol and return the final GraphState."""
    if seed_node is None:
        seed_node = random.randrange(full_graph.num_nodes())
    reader.reset()
    state = GraphState(full_graph, seed_node=seed_node)
    loop = RevealLoop(state, reader)
    loop.run(budget=budget)
    return state


def reveal_one(
    reader: BaseReader,
    full_graph: dgl.DGLGraph,
    budget: int,
    seed_node: Optional[int] = None,
) -> Tuple[dgl.DGLGraph, List[int]]:
    """Reveal one graph and build the final subgraph (revealed nodes/edges only)."""
    from ..gnn.subgraph_builder import SubgraphBuilder
    builder = SubgraphBuilder()
    state = run_reveal(reader, full_graph, budget, seed_node)
    return builder.build(state)


def reveal_batch(
    reader: BaseReader,
    full_graphs: List[dgl.DGLGraph],
    budgets: List[int],
    seed_nodes: Optional[List[int]] = None,
) -> Tuple[dgl.DGLGraph, List[List[int]]]:
    """Reveal each graph, then batch them."""
    from ..gnn.subgraph_builder import SubgraphBuilder
    if seed_nodes is None:
        seed_nodes = [random.randrange(g.num_nodes()) for g in full_graphs]

    builder = SubgraphBuilder()
    states = []
    for g, b, s in zip(full_graphs, budgets, seed_nodes):
        state = run_reveal(reader, g, b, s)
        states.append(state)

    return builder.build_batch(states)
