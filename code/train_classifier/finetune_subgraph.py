"""Subgraph-adapted classifier fine-tuning (any registered dataset).

A full-graph classifier can collapse out-of-distribution on partially
revealed subgraphs (e.g. COLLAB without node features), removing evaluation
discrimination. This script fine-tunes the judge with protocol-consistent
subgraph sampling (fixed seed formula + frontier-constrained reveal + spec
budget tiers), mixing in a fraction of full graphs (--full-frac) to prevent
forgetting the full-graph reference. Sampling strategies (random /
random-walk / bfs) rotate per graph; every epoch resamples with a new seed_i
while validation uses a fixed seed for comparability. Structure is inherited
from --base-ckpt (required); checkpoints embed the full config; model
selection uses the mean subgraph accuracy over budget tiers on validation."""

import argparse
import datetime
import random
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))
sys.path = [p for p in sys.path
            if Path(p).resolve() != Path(__file__).resolve().parent]

from tools.paths import CODE_DIR, REPO_ROOT  # noqa: E402
from train_classifier.class_weight import compute_class_weight  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
import dgl  # noqa: E402
from tqdm import tqdm  # noqa: E402

from classifier import GraphClassifier  # noqa: E402
from classifier.factory import classifier_config_from_ckpt  # noqa: E402
from datasets import get_dataset  # noqa: E402
from tools.config import build_experiment_config, load_json, save_json  # noqa: E402
from tools.seeds import deterministic_seed, set_all_seeds  # noqa: E402

from active_graph_reader.engine.graph_state import GraphState  # noqa: E402
from active_graph_reader.engine.reveal_action import reveal_node  # noqa: E402
from active_graph_reader.reader.fixed import create_fixed_reader  # noqa: E402
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder  # noqa: E402


def resolve_path(p: str) -> Path:
    from tools.paths import resolve as _resolve
    return _resolve(p)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='subgraph-adapted classifier fine-tuning')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--root', required=True, help='dataset root')
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='max graphs per split (0 = all)')
    parser.add_argument('--base-ckpt', required=True,
                        help='full-graph classifier checkpoint to fine-tune (structure inherited)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='fine-tune learning rate (default 1e-4)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--class-weight', default='none',
                        choices=['none', 'balanced'],
                        help='cross_entropy class weights (as train_full_graph); use balanced '
                             'on majority-class collapse (default none)')
    parser.add_argument('--policies', nargs='+',
                        default=['random', 'random-walk', 'bfs'],
                        choices=['random', 'random-walk', 'bfs'],
                        help='subgraph sampling strategy mix (rotates per graph)')
    parser.add_argument('--budget-ratios', type=float, nargs='+', default=None,
                        help='train/val budget ratios (default spec.budget_options)')
    parser.add_argument('--subgraphs-per-graph', type=int, default=1,
                        help='subgraphs sampled per graph per epoch')
    parser.add_argument('--full-frac', type=float, default=0.1,
                        help='fraction of training samples kept as full graphs (anti-forgetting)')
    parser.add_argument('--val-max-graphs', type=int, default=1000,
                        help='max validation graphs (0 = all; caps per-epoch validation cost)')
    parser.add_argument('--eval-every', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', default='',
                        help='experiment config JSON (explicit CLI args override)')
    parser.add_argument('--output-dir', default='',
                        help='default result/{dataset}/classifier/finetune_subgraph')
    return parser


def _ensure_policies_registered(policy_names):
    if 'random-walk' in policy_names:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        import baseline.strategies
        baseline.strategies.register_baseline_strategies()


def make_agents(policy_names):
    _ensure_policies_registered(policy_names)
    return {p: create_fixed_reader(p) for p in policy_names}


def rollout_partial(graph, agent, budget_ratio, graph_idx: int, seed_i: int,
                    builder: SubgraphBuilder):
    """Sample one partial subgraph under the protocol (same logic as the eval
    fixed-strategy rollout).

    Budget = revealed nodes excluding the seed; the final reveal that
    reaches the budget is included in the trajectory.
    """
    state = GraphState(graph, deterministic_seed(graph_idx, seed_i,
                                                 graph.num_nodes()))
    state._budget_limit = max(1, int(budget_ratio * state.n_nodes_total))
    agent.reset()
    if state._initial_frontier:
        agent.on_new_frontier(list(state._initial_frontier))
    while not state.is_frontier_empty and state.get_budget_used() < state._budget_limit:
        v = agent.select(state)
        if v is None:
            break
        new_frontier = reveal_node(state, v)
        if new_frontier:
            agent.on_new_frontier(new_frontier)
    return builder.build(state)[0]


def train_one_epoch(clf, optimizer, graphs, labels, args, policies, agents,
                    ratios, builder, epoch: int, device, rng,
                    class_weight=None):
    clf.train()
    order = list(range(len(graphs)))
    rng.shuffle(order)
    total_loss = n_batches = correct = total = 0
    for start in tqdm(range(0, len(order), args.batch_size), desc='  Train'):
        idx = order[start:start + args.batch_size]
        sub_graphs = []
        ys = []
        for pos in idx:
            if args.full_frac > 0 and rng.random() < args.full_frac:
                sub_graphs.append(graphs[pos])
                ys.append(int(labels[pos]))
                continue
            k = rng.randrange(args.subgraphs_per_graph)
            policy = policies[(pos * args.subgraphs_per_graph + k) % len(policies)]
            ratio = ratios[(pos * args.subgraphs_per_graph + k) % len(ratios)]
            sub_g = rollout_partial(graphs[pos], agents[policy], ratio,
                                    graph_idx=pos,
                                    seed_i=epoch * args.subgraphs_per_graph + k,
                                    builder=builder)
            sub_graphs.append(sub_g)
            ys.append(int(labels[pos]))
        batch_g = dgl.batch(sub_graphs).to(device)
        nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        y = torch.tensor(ys, dtype=torch.long, device=device)
        logits = clf(batch_g, nids)
        loss = nn.functional.cross_entropy(logits, y, weight=class_weight,
                                           label_smoothing=args.label_smoothing)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += len(ys)
    return total_loss / max(n_batches, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate_subgraphs(clf, graphs, labels, policies, agents, ratios,
                       builder, device, batch_size=256, val_max=0):
    """Validation subgraph accuracy: one fixed-seed subgraph per (graph,
    ratio), strategies rotating by (graph_pos + ratio_idx). Returns
    (per_ratio dict, mean)."""
    clf.eval()
    n = len(graphs) if val_max <= 0 else min(val_max, len(graphs))
    per_ratio = {f'{r:.2f}': [0, 0] for r in ratios}
    for r_idx, ratio in enumerate(ratios):
        for start in range(0, n, batch_size):
            pos = list(range(start, min(start + batch_size, n)))
            subs, ys = [], []
            for p in pos:
                policy = policies[(p + r_idx) % len(policies)]
                sub = rollout_partial(graphs[p], agents[policy], ratio,
                                      graph_idx=p, seed_i=10_000 + p * len(ratios) + r_idx,
                                      builder=builder)
                subs.append(sub)
                ys.append(int(labels[p]))
            batch_g = dgl.batch(subs).to(device)
            nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
            pred = clf(batch_g, nids).argmax(dim=-1).cpu()
            y = torch.tensor(ys, dtype=torch.long)
            per_ratio[f'{ratio:.2f}'][0] += (pred == y).sum().item()
            per_ratio[f'{ratio:.2f}'][1] += len(ys)
    accs = {k: c / max(t, 1) for k, (c, t) in per_ratio.items()}
    return accs, sum(accs.values()) / max(len(accs), 1)


@torch.no_grad()
def evaluate_full(clf, graphs, labels, device, batch_size=256, val_max=0):
    clf.eval()
    n = len(graphs) if val_max <= 0 else min(val_max, len(graphs))
    correct = total = 0
    for start in range(0, n, batch_size):
        idx = list(range(start, min(start + batch_size, n)))
        batch_g = dgl.batch([graphs[i] for i in idx]).to(device)
        nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        pred = clf(batch_g, nids).argmax(dim=-1).cpu()
        correct += (pred == labels[idx]).sum().item()
        total += len(idx)
    return correct / max(total, 1)


def _strip_dgl_id(graphs):
    """Strip the TUDataset _ID frame column.

    SubgraphBuilder products lack it; mixing full graphs and subgraphs in one
    batch would make dgl.batch fail on the schema mismatch (_ID is an internal
    bookkeeping column, not a model input)."""
    for g in graphs:
        if '_ID' in g.ndata:
            g.ndata.pop('_ID')
        if '_ID' in g.edata:
            g.edata.pop('_ID')


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.config:
        cfg_baseline = load_json(args.config)
        for k, v in cfg_baseline.items():
            if hasattr(args, k) and getattr(args, k) == parser.get_default(k):
                setattr(args, k, v)

    set_all_seeds(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ds = get_dataset(args.dataset, root=args.root, max_graphs=args.max_graphs)
    spec = ds.spec
    ratios = list(args.budget_ratios) if args.budget_ratios \
        else list(spec.budget_options)
    print(f"Dataset {spec.name}: {spec.n_classes} classes, "
          f"budget_ratios={ratios}, policies={args.policies}")

    train_graphs, train_labels = ds.train_graphs, ds.train_labels
    valid_graphs = ds.valid_graphs
    valid_labels = ds.valid_labels
    if not valid_graphs:
        print("[warn] no validation split; model selection falls back to training subgraph accuracy")
        valid_graphs, valid_labels = train_graphs, train_labels
    _strip_dgl_id(train_graphs)
    _strip_dgl_id(valid_graphs)
    print(f"Train: {len(train_graphs)} graphs | Valid: {len(valid_graphs)} graphs")

    class_weight = compute_class_weight(args.class_weight, train_labels,
                                        spec.n_classes, device=device)

    base_path = resolve_path(args.base_ckpt)
    base_ckpt = torch.load(str(base_path), map_location=device, weights_only=False)
    clf_cfg = classifier_config_from_ckpt(base_ckpt)
    clf = GraphClassifier(config=clf_cfg).to(device)
    clf.load_state_dict(base_ckpt['model_state_dict'], strict=True)
    print(f"Loaded base weights from {base_path} | "
          f"params={sum(p.numel() for p in clf.parameters()):,}")

    agents = make_agents(args.policies)
    builder = SubgraphBuilder()
    optimizer = optim.Adam(clf.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    out_dir = resolve_path(args.output_dir) if args.output_dir \
        else CODE_DIR / 'result' / spec.name / 'classifier' / 'finetune_subgraph'
    out_dir.mkdir(parents=True, exist_ok=True)
    last_path = out_dir / 'subgraph_classifier.pt'
    best_path = out_dir / 'best_subgraph_classifier.pt'

    experiment_cfg = build_experiment_config(args, extra={
        'classifier': clf_cfg.to_dict(),
        'base_checkpoint': str(base_path),
        'budget_ratios': ratios,
        'device': str(device),
        'started_at': datetime.datetime.now().isoformat(timespec='seconds'),
    })
    save_json(experiment_cfg, out_dir / 'config.json')

    ckpt_config = {'classifier': clf_cfg.to_dict(), 'dataset': spec.name,
                   'n_classes': spec.n_classes,
                   'judge_kind': 'subgraph-finetuned',
                   'base_checkpoint': str(base_path),
                   'policies': list(args.policies),
                   'budget_ratios': ratios,
                   'full_frac': args.full_frac}

    history = {'dataset': spec.name, 'base_checkpoint': str(base_path),
               'budget_ratios': ratios, 'epochs': []}
    best_mean, best_epoch = -1.0, None
    rng = random.Random(args.seed)

    for epoch in range(1, args.epochs + 1):
        loss, train_acc = train_one_epoch(clf, optimizer, train_graphs,
                                          train_labels, args, args.policies,
                                          agents, ratios, builder, epoch,
                                          device, rng,
                                          class_weight=class_weight)
        rec = {'epoch': epoch, 'loss': round(loss, 6),
               'train_acc': round(train_acc, 6)}
        line = f"Epoch {epoch}/{args.epochs}: loss={loss:.4f} train_acc={train_acc:.4f}"

        if args.eval_every > 0 and epoch % args.eval_every == 0:
            sub_accs, sub_mean = evaluate_subgraphs(
                clf, valid_graphs, valid_labels, args.policies, agents,
                ratios, builder, device, val_max=args.val_max_graphs)
            full_acc = evaluate_full(clf, valid_graphs, valid_labels, device,
                                     val_max=args.val_max_graphs)
            rec.update({'val_sub_acc': sub_accs, 'val_sub_acc_mean': round(sub_mean, 6),
                        'val_full_acc': round(full_acc, 6)})
            line += (f"  val_sub={sub_mean:.4f} "
                     f"({', '.join(f'{k}={v:.3f}' for k, v in sub_accs.items())})  "
                     f"val_full={full_acc:.4f}")
            ckpt = {'model_state_dict': clf.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch, 'config': ckpt_config,
                    'val_acc': round(sub_mean, 6),
                    'val_acc_per_ratio': sub_accs,
                    'val_full_acc': round(full_acc, 6)}
            torch.save(ckpt, str(last_path))
            if sub_mean > best_mean:
                best_mean, best_epoch = sub_mean, epoch
                ckpt['best_acc'], ckpt['best_epoch'] = best_mean, best_epoch
                torch.save(ckpt, str(best_path))
                line += f"  [best] → {best_path.name}"
        history['epochs'].append(rec)
        history['best'] = (None if best_epoch is None
                           else {'epoch': best_epoch, 'val_sub_acc_mean': round(best_mean, 6),
                                 'checkpoint': best_path.name})
        save_json(history, out_dir / 'history.json')
        print(line)

    print(f"Done. Best val_sub_acc={best_mean:.4f} @ epoch {best_epoch} → {best_path}")
    if best_epoch is not None:
        ck = torch.load(str(best_path), map_location='cpu', weights_only=False)
        print(f"  val_acc_per_ratio={ck['val_acc_per_ratio']}  "
              f"val_full_acc={ck['val_full_acc']}")


if __name__ == '__main__':
    main()
