"""Teacher trajectory generation for any registered dataset.

Teachers come from trajectories.teachers.build_teacher (gradient /
structural-* / info-gain / hybrid). --labels restricts generation to given
classes; --k-shards/--shard-idx write trajectories_r{key}_shard{i}of{K}.pt
and skip existing shards, so re-running the same command resumes after an
interruption. graph_idx = local position in ds.train_graphs (consistent with
train_bc/train_rl); with --stratified --max-graphs, original indices are
recorded (train_indices aligns them in training)."""

import argparse
import json
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))
sys.path = [p for p in sys.path
            if Path(p).resolve() != Path(__file__).resolve().parent]

import torch  # noqa: E402

from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder  # noqa: E402
from datasets import get_dataset  # noqa: E402
from tools.paths import CODE_DIR  # noqa: E402
from trajectories.core.generator import TrajectoryGenerator  # noqa: E402
from trajectories.core.rollout import build_classifier  # noqa: E402
from trajectories.teachers import build_teacher  # noqa: E402


class _GraphList:
    """Wrap train_graphs + train_labels as dataset[i] -> (g, lbl) for generate_all."""

    def __init__(self, graphs, labels):
        self._graphs = graphs
        self._labels = labels

    def __len__(self):
        return len(self._graphs)

    def __getitem__(self, i):
        return self._graphs[i], self._labels[i]


def ratio_key(r: float) -> str:
    """0.05 -> '0_050'."""
    return f'{r:.3f}'.replace('.', '_')


def main():
    parser = argparse.ArgumentParser(description='teacher trajectory generation')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--root', default='',
                        help='dataset root (e.g. D:\\superpixels)')
    parser.add_argument('--classifier', default='',
                        help='upper-bound classifier checkpoint (absolute or relative to code/)')
    parser.add_argument('--teachers', nargs='+',
                        default=['gradient', 'structural-degree'])
    parser.add_argument('--budget-ratios', type=float, nargs='+', default=None,
                        help='budget ratios (default spec.budget_options)')
    parser.add_argument('--n-seeds', type=int, default=5,
                        help='top-n seeds per graph per teacher')
    parser.add_argument('--labels', type=int, nargs='+', default=None,
                        help='only generate trajectories for these classes (default all)')
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='training graph limit (0 = all; small value for smoke tests)')
    parser.add_argument('--stratified', action='store_true',
                        help='stratify --max-graphs sampling by class (default: first N; '
                             'trajectories record the original indices)')
    parser.add_argument('--k-shards', type=int, default=1,
                        help='split the graph list into K shards (resume unit)')
    parser.add_argument('--shard-idx', type=int, nargs='+', default=None,
                        help='shard ids to run (default all; existing shard files are skipped)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--graph-batch-size', type=int, default=32)
    parser.add_argument('--output-dir', default='',
                        help='output directory (default data/{dataset}/trajectories)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', default='',
                        help='experiment config JSON (keys = underscored CLI names; explicit '
                             'CLI args override)')
    args = parser.parse_args()

    cfg_baseline = {}
    if args.config:
        from tools.config import load_json
        cfg_baseline = load_json(args.config)
        for k, v in cfg_baseline.items():
            if hasattr(args, k) and getattr(args, k) == parser.get_default(k):
                setattr(args, k, v)
    from tools import cli as cli_utils
    cli_utils.attach_defaults(args, parser)
    from tools.config import require_args
    require_args(args, ['root', 'classifier'])

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ds = get_dataset(args.dataset, root=args.root or None, splits=['train'],
                     max_graphs=args.max_graphs,
                     stratified=args.stratified and args.max_graphs > 0)
    spec = ds.spec
    ratios = args.budget_ratios or list(spec.budget_options)

    train_labels = ds.train_labels
    if args.labels:
        keep = [i for i in range(len(train_labels))
                if int(train_labels[i]) in set(args.labels)]
        print(f"Label filter {args.labels}: {len(keep)}/{len(train_labels)} graphs")
    else:
        keep = list(range(len(train_labels)))

    output_dir = (Path(args.output_dir) if args.output_dir
                  else CODE_DIR / 'data' / spec.name / 'trajectories')
    if not output_dir.is_absolute():
        output_dir = CODE_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    k = max(1, args.k_shards)
    shard_ids = args.shard_idx if args.shard_idx is not None else list(range(k))
    shards = []            # (shard_id, [graph_idx...], output_suffix)
    for sid in shard_ids:
        part = keep[sid::k] if k > 1 else keep
        suffix = f'_shard{sid}of{k}' if k > 1 else ''
        missing = {r: output_dir / f'trajectories_r{ratio_key(r)}{suffix}.pt'
                   for r in ratios}
        if all(p.exists() for p in missing.values()):
            print(f"Shard {sid}: all ratios exist, skipping (resume)")
            continue
        shards.append((sid, part, suffix))
    if not shards:
        print("All shards already complete; nothing to generate.")
        return

    from tools.paths import resolve as _resolve
    clf_path = _resolve(args.classifier)
    cls_ckpt = torch.load(str(clf_path), map_location=device, weights_only=False)
    classifier = build_classifier(cls_ckpt, device, requires_grad=True)
    print(f"Classifier: {clf_path} (edge_feat_dim={spec.edge_feat_dim}, "
          f"n_classes={spec.n_classes})")

    builder = SubgraphBuilder()
    teachers = [build_teacher(name, classifier=classifier, subgraph_builder=builder,
                              device=str(device)) for name in args.teachers]
    print(f"Teachers: {[t.name for t in teachers]}")

    generator = TrajectoryGenerator(
        teachers=teachers,
        classifier=classifier,
        subgraph_builder=builder,
        device=str(device),
        graph_batch_size=args.graph_batch_size,
    )

    summary = {}
    for sid, part, suffix in shards:
        dataset = _GraphList([ds.train_graphs[i] for i in part],
                             [train_labels[i] for i in part])
        if args.stratified and args.max_graphs > 0:
            idx_part = [ds.train_indices[i] for i in part]
        else:
            idx_part = part
        for ratio in ratios:
            key = ratio_key(ratio)
            out_path = output_dir / f'trajectories_r{key}{suffix}.pt'
            if out_path.exists():
                print(f"  shard{sid} r={ratio}: exists, skipping")
                continue
            print(f"\n=== shard{sid} reveal_ratio={ratio} ({len(part)} graphs) ===")
            result = generator.generate_all(dataset, idx_part, ratio,
                                            n_seeds=args.n_seeds)
            n_correct = {t: sum(len(l) for l in result[t])
                         for t in result}
            summary[f'shard{sid}_{key}'] = n_correct
            for t, cnt in n_correct.items():
                print(f"  {t}: {cnt} correct trajectories")
            torch.save({'trajectories': result, 'config': {
                'dataset': args.dataset,
                'budget_ratio': ratio,
                'teachers': list(result.keys()),
                'n_seeds': args.n_seeds,
                'n_graphs': len(part),
                'labels': args.labels,
                'shard': {'idx': sid, 'k': k} if k > 1 else None,
            }}, str(out_path))
            print(f"  Saved: {out_path}")

    print("\n=== Yield summary (shard x ratio) ===")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
