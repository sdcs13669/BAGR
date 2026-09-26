"""Graph readout pooling: encoder final-layer node representations -> graph
representation.

- mean: AvgPooling (default; no params, legacy-compatible keys);
- meanmax: concat of mean and max, doubling out_dim. Neither mean nor per-dim
  max scales with node count, so readout magnitudes match between full-graph
  training and small revealed-subgraph evaluation (subgraph robustness; this
  is why a node-count-scaling sum is not offered here).

The GNN+ paper uses single poolings (mean on CIFAR10-SP/OGB, sum on ZINC, max
on MalNet-Tiny); meanmax is a combination variant for subgraph robustness."""

import dgl.nn as dglnn

READOUT_KINDS = ('mean', 'meanmax', 'sum')
READOUT_OUT_MULT = {'mean': 1, 'meanmax': 2, 'sum': 1}


def make_readout(kind: str):
    """kind -> (pooling modules, out_mult); meanmax returns (avg, max)."""
    if kind == 'mean':
        return (dglnn.AvgPooling(),), 1
    if kind == 'meanmax':
        return (dglnn.AvgPooling(), dglnn.MaxPooling()), 2
    if kind == 'sum':
        return (dglnn.SumPooling(),), 1
    raise ValueError(f"unknown readout pooling '{kind}', choose: {READOUT_KINDS}")
