# BAGR: Budget-Aware Graph Reader (anonymous code)

Code for the paper "Learning Where to Read: Graph Classification under
Observation Budgets" (double-blind submission). BAGR sequentially reveals
nodes under a node budget and classifies the revealed subgraph. It is trained
in two stages: behavior cloning (BC) on filtered teacher trajectories, then
REINFORCE fine-tuning with a reward judged by a frozen upper-bound classifier
(warm-up on the classifier-only judgment, then the combined classifier +
evidence-head judgment).

## Layout

```
code/
  active_graph_reader/   core: GraphState/reveal engine, evidence reader,
                         fixed strategies (random / bfs / max-frontier-degree)
  classifier/            frozen upper-bound classifier (GINE backbone)
  datasets/              CIFAR10 superpixels / COLLAB / OGBG-PPA loaders
  trajectories/          teacher trajectory generation + filtering + balancing
  train_classifier/      upper-bound classifier training (+ subgraph fine-tune)
  train_reader/          BC and REINFORCE training, reader/baseline evaluation
  tools/                 seeds, config merge, checkpoint IO, path bootstrap
baseline/                protocol-adapted baselines (random-walk / SAGPool /
                         Soft-Mask / AgentNet) with training scripts
```

## Environment

Python 3.10 with `torch 2.3.0+cu121`, `dgl 2.2.1`, `ogb 1.3.6`,
`torch-geometric 2.7.0`, `scikit-learn`. GPU strongly recommended.

One-shot setup on a Linux server with conda installed:

```bash
bash setup_env.sh    # creates the env from code/environment-linux.yml and verifies imports
conda activate gnn
```

or manually: `conda env create -f code/environment-linux.yml`.

## Reproducing the pipeline

All scripts run from this directory (`code/` is put on `sys.path`
automatically). Example on CIFAR10 superpixels:

```bash
# 1. upper-bound classifier (full graphs)
python code/train_classifier/train_full_graph.py --dataset cifar10 \
    --root D:/data/superpixels --epochs 30

# 2. teacher trajectories (gradient + structural teachers)
python code/trajectories/generate_trajectories.py --dataset cifar10 \
    --root D:/data/superpixels --classifier result/cifar10/classifier/<ckpt>.pt \
    --teachers gradient structural-degree

# 3. filter: similarity dedup + per-graph top-k + teacher diversity
python code/trajectories/filter_trajectories.py \
    --input-dir data/cifar10/trajectories --output-dir data/cifar10/trajectories_filtered

# 4. behavior cloning (evidence reader)
python code/train_reader/train_bc.py --dataset cifar10 \
    --root D:/data/superpixels --data-dir data/cifar10/trajectories_filtered \
    --reader-kind evidence --epochs 10

# 5. REINFORCE fine-tuning (BC warm start, classifier-judged reward)
python code/train_reader/train_rl.py --dataset cifar10 \
    --root D:/data/superpixels --reader-kind evidence \
    --bootstrap <bc checkpoint>.pt --classifier <classifier>.pt \
    --reward asymmetric

# 6. evaluation (fixed strategies + baselines / trained reader)
python code/train_reader/eval_baselines.py --dataset cifar10 \
    --root D:/data/superpixels --strategy random bfs --classifier <classifier>.pt
python code/train_reader/eval_reader.py --dataset cifar10 \
    --root D:/data/superpixels --checkpoint <rl checkpoint>.pt \
    --classifier <classifier>.pt
```

Protocol: the budget is the number of revealed nodes excluding the seed
(5/10/15% of nodes for CIFAR10/PPA, 10/15/20% for COLLAB); the seed node is a
deterministic function of the graph index; frontier edge features are visible
before reveal at zero budget cost; every evaluation is deterministic (argmax,
ties to the smaller node id, four seeds per graph averaged).

Baselines (need a scorer checkpoint trained on full graphs first):

```bash
python baseline/run_sagpool.py --dataset cifar10 --root D:/data/superpixels
python code/train_reader/eval_baselines.py --dataset cifar10 \
    --root D:/data/superpixels --strategy sagpool \
    --strategy-ckpt baseline/artifacts/cifar10/sagpool.pt \
    --classifier <classifier>.pt
```

PPA note: the full OGBG-PPA train split is ~158k graphs; always pass
`--max-graphs` (evaluating readers/baselines on PPA also needs a small
`--chunk-size`/reference batch on memory-limited GPUs).
