"""Behavior cloning (BC) training for the evidence reader.

Trajectory data is produced by trajectories/generate_trajectories.py
(+ filter/balance). Effective config is written to <output>/config.json and
embedded in every checkpoint; --resume continues from a checkpoint config
(--epochs is the total). Output: result/{dataset}/experiments/{exp}/evidence/bc/.
"""

import argparse
import datetime
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))
sys.path = [p for p in sys.path
            if Path(p).resolve() != Path(__file__).resolve().parent]

import torch  # noqa: E402

from active_graph_reader import ReaderConfig, build_reader  # noqa: E402
from datasets import get_dataset  # noqa: E402
from tools import cli as cli_utils  # noqa: E402
from tools.ckpt_io import load_optimizer, load_training_state, save_epoch_checkpoint  # noqa: E402
from tools.config import build_experiment_config, load_json, save_json  # noqa: E402
from tools.paths import CODE_DIR  # noqa: E402
from tools.seeds import set_all_seeds  # noqa: E402
from tools.trajectory_io import (load_trajectory_file,  # noqa: E402
                                 normalize_trajectory)
from train_reader.core import bc_train_epoch  # noqa: E402


def resolve(p) -> Path:
    from tools.paths import resolve as _resolve
    return _resolve(p)


def merge_trajectories(data_dir: Path, teachers: list) -> list:
    """Load trajectories_r*.pt files in a directory, filter by teacher, flatten."""
    all_trajs = []
    for fpath in sorted(data_dir.glob('trajectories_r*.pt')):
        data = load_trajectory_file(fpath)
        n_before = len(all_trajs)
        for teacher_name in teachers:
            for per_graph_list in data['trajectories'].get(teacher_name, []):
                for t in per_graph_list:
                    all_trajs.append(normalize_trajectory(t))
        print(f"  {fpath.name}: +{len(all_trajs) - n_before} trajectories")
    print(f"Total merged: {len(all_trajs)} trajectories")
    return all_trajs


def build_graph_cache(spec, ds, needed: set) -> dict:
    """graph_idx -> (graph, label); PPA loads selectively via load_graph_subset.

    With a stratified subset, graph_idx indexes the original split and is
    aligned through train_indices (loader guarantees same ordering)."""
    if spec.load_graph_subset is not None:
        return spec.load_graph_subset(spec, needed)
    if getattr(spec, 'stratified', False) and spec.max_graphs > 0:
        return {gid: (ds.train_graphs[pos], int(ds.train_labels[pos]))
                for pos, gid in enumerate(ds.train_indices) if gid in needed}
    return {i: (ds.train_graphs[i], int(ds.train_labels[i])) for i in needed}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='BC training (evidence reader)')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--root', default='', help='dataset root (required: CLI or --config)')
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='max graphs per split (0 = no limit)')
    parser.add_argument('--stratified', action='store_true',
                        help='stratify --max-graphs sampling by class (default: first N; '
                             'trajectories filtered by subset graph_idx)')
    parser.add_argument('--data-dir', default='',
                        help='trajectory directory (default data/{dataset}/trajectories)')
    parser.add_argument('--teachers', nargs='+',
                        default=['gradient', 'structural-degree'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=32)
    cli_utils.add_reader_args(parser, reader_kind_default='evidence')
    parser.add_argument('--evidence-lambda', type=float, default=1.0,
                        help='weight of L_ev for the evidence reader (L = L_act + lambda*L_ev)')
    parser.add_argument('--edge-feat-dim', type=int, default=None,
                        help='override dataset edge feature dim (default: DatasetSpec)')
    parser.add_argument('--node-feat-dim', type=int, default=None,
                        help='override dataset node feature dim (default: DatasetSpec)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--exp', default='default',
                        help='experiment name (output result/{ds}/experiments/{exp}/)')
    parser.add_argument('--output-dir', default='',
                        help='override the default output directory')
    parser.add_argument('--resume', default='',
                        help='resume from a BC checkpoint; structure/hparams from its config')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', default='', help='experiment config JSON')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg_baseline = load_json(args.config) if args.config else {}
    if cfg_baseline:
        for k, v in cfg_baseline.items():
            if hasattr(args, k) and getattr(args, k) == parser.get_default(k):
                setattr(args, k, v)
    cli_utils.attach_defaults(args, parser)
    from tools.config import require_args
    require_args(args, ['root'])

    set_all_seeds(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ds = get_dataset(args.dataset, root=args.root, splits=['train'],
                     max_graphs=args.max_graphs,
                     stratified=getattr(args, 'stratified', False))
    spec = ds.spec

    data_dir = resolve(args.data_dir) if args.data_dir else \
        CODE_DIR / 'data' / spec.name / 'trajectories'
    print("\n=== Merging trajectories ===")
    all_trajs = merge_trajectories(data_dir, args.teachers)
    if not all_trajs:
        raise SystemExit(f"no trajectories in {data_dir}; run trajectories/generate_trajectories.py first")
    if args.max_graphs > 0 and getattr(args, 'stratified', False):
        allowed = set(ds.train_indices)
        n0 = len(all_trajs)
        all_trajs = [t for t in all_trajs if t.graph_idx in allowed]
        print(f"  stratified subset: {n0} -> {len(all_trajs)} trajectories"
              f" ({len(allowed)} graphs in the stratified subset)")

    print("\n=== Building graph cache ===")
    graph_cache = build_graph_cache(spec, ds, {t.graph_idx for t in all_trajs})
    print(f"  {len(graph_cache)} graphs cached")

    start_epoch, training_log = 1, []
    resume_ckpt = None
    if args.resume:
        resume_path = resolve(args.resume)
        resume_ckpt = torch.load(str(resume_path), map_location='cpu',
                                 weights_only=False)
        from train_reader.core import load_reader_from_checkpoint
        reader = load_reader_from_checkpoint(resume_ckpt, device)
        config = (resume_ckpt.get('config')
                  if isinstance(resume_ckpt.get('config'), dict) else None) or {}
        optimizer = torch.optim.Adam(reader.parameters(),
                                     lr=config.get('lr', args.lr))
        load_optimizer(optimizer, resume_ckpt, device=str(device))
        start_epoch, training_log = load_training_state(resume_ckpt)
        if start_epoch > args.epochs:
            raise SystemExit(f"checkpoint already finished {start_epoch - 1} epochs; "
                             f"--epochs={args.epochs} leaves nothing to train")
        print(f"Resuming from {resume_path} (done={start_epoch - 1}, "
              f"target={args.epochs})")
    else:
        cfg_dict = cli_utils.reader_config_from_args(args)
        if cfg_dict.get('kind') == 'evidence':
            cfg_dict['n_classes'] = spec.n_classes
        reader_cfg = ReaderConfig.from_dict(cfg_dict)
        reader_cfg.edge_feat_dim = args.edge_feat_dim or spec.edge_feat_dim
        reader_cfg.node_feat_dim = args.node_feat_dim or spec.node_feat_dim
        reader_cfg.validate()
        reader = build_reader(reader_cfg).to(device)
        optimizer = torch.optim.Adam(reader.parameters(), lr=args.lr)

    total_params = sum(p.numel() for p in reader.parameters())
    print(f"Reader: {reader.config.kind}, params={total_params:,}, "
          f"edge_feat_dim={reader.config.edge_feat_dim}, "
          f"node_feat_dim={reader.config.node_feat_dim}")

    output_dir = (resolve(args.output_dir) if args.output_dir else
                  CODE_DIR / 'result' / spec.name / 'experiments' / args.exp /
                  reader.config.kind / 'bc')
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and isinstance(resume_ckpt.get('config'), dict) \
            and resume_ckpt['config'].get('reader'):
        config = resume_ckpt['config']
        config['epochs'] = args.epochs
    else:
        config = build_experiment_config(args, extra={
            'reader': reader.config.to_dict(),
            'dataset': spec.name,
            'dataset_spec': {'n_classes': spec.n_classes,
                             'node_feat_dim': spec.node_feat_dim,
                             'edge_feat_dim': spec.edge_feat_dim,
                             'budget_options': list(spec.budget_options)},
            'epochs': args.epochs,
            'total_params': total_params,
            'n_train_trajs': len(all_trajs),
            'data_dir': str(data_dir),
            'started_at': datetime.datetime.now().isoformat(timespec='seconds'),
        })
    save_json(config, output_dir / 'config.json')
    print(f"Config saved: {output_dir / 'config.json'}")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        metrics = bc_train_epoch(reader, optimizer, graph_cache, all_trajs,
                                 args.batch_size, device,
                                 temperature=1.0,
                                 evidence_lambda=args.evidence_lambda)
        print(f"  Train loss: {metrics['loss']:.4f}"
              f", act_acc: {metrics['act_acc']:.4f}"
              + (f", ev_acc: {metrics['ev_acc']:.4f}"
                 if 'ev_acc' in metrics else '')
              + (f", ev_acc_final: {metrics['ev_acc_final']:.4f}"
                 if 'ev_acc_final' in metrics else ''))
        log_entry = {'epoch': epoch, 'train_loss': metrics['loss'],
                     'act_acc': metrics['act_acc']}
        if 'ev_acc' in metrics:
            log_entry['ev_acc'] = metrics['ev_acc']
        if 'ev_acc_final' in metrics:
            log_entry['ev_acc_final'] = metrics['ev_acc_final']
        training_log.append(log_entry)

        save_epoch_checkpoint(
            output_dir / f'reader_checkpoint_epoch{epoch}.pt',
            epoch=epoch, reader=reader, optimizer=optimizer,
            config=config, training_log=training_log)
        save_json(training_log, output_dir / 'training_log.json')
        print(f"  Saved: reader_checkpoint_epoch{epoch}.pt + training_log.json")

    print(f"\nBC complete. Output: {output_dir}")


if __name__ == '__main__':
    main()
