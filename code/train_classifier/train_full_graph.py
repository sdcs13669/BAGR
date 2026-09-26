"""Full-graph upper-bound classifier training (any registered dataset).

Structure comes from --hidden-dims per-layer lists or the uniform
--hidden-dim/--n-layers/--dropout form; --config experiment.json is also
supported with explicit CLI overrides. Checkpoints embed the full config
(ClassifierConfig + dataset spec) so classifier.factory.load_classifier can
rebuild without hardcoded dims. --resume restores model/optimizer/scheduler/
epoch/best_acc/history. Output defaults to result/{dataset}/classifier/."""

import argparse
import datetime
import random
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))
sys.path = [p for p in sys.path
            if Path(p).resolve() != Path(__file__).resolve().parent]

from tools.paths import CODE_DIR  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
import dgl  # noqa: E402
from tqdm import tqdm  # noqa: E402

from classifier import GraphClassifier, ClassifierConfig  # noqa: E402
from datasets import get_dataset  # noqa: E402
from tools.config import (build_experiment_config, load_json,  # noqa: E402
                          merge_args_with_config, save_json)
from tools.seeds import set_all_seeds  # noqa: E402
from train_classifier.class_weight import compute_class_weight  # noqa: E402


def resolve_path(p: str) -> Path:
    from tools.paths import resolve as _resolve
    return _resolve(p)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='full-graph upper-bound classifier training')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--root', default='',
                        help='dataset root (required: CLI or --config)')
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='max graphs per split (0 = all)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--class-weight', default='none',
                        choices=['none', 'balanced'],
                        help='cross_entropy class weights: use balanced when a majority '
                             'class exceeds 50 percent (w_c=N/(K*count_c); '
                             'default none = unweighted)')
    parser.add_argument('--scheduler-patience', type=int, default=0,
                        help='ReduceLROnPlateau patience (0 = no scheduler)')
    parser.add_argument('--early-stop-patience', type=int, default=0,
                        help='stop after N epochs without test-acc improvement (0 = off)')
    parser.add_argument('--eval-every', type=int, default=5,
                        help='evaluate on test every N epochs and keep that checkpoint (0 = off)')
    parser.add_argument('--hidden-dim', type=int, default=300)
    parser.add_argument('--n-layers', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=None,
                        help='per-layer width list (overrides the uniform form)')
    parser.add_argument('--dropouts', type=float, nargs='+', default=None,
                        help='per-layer dropout list')
    parser.add_argument('--virtual-node', action='store_true')
    parser.add_argument('--residual', action='store_true')
    parser.add_argument('--backbone', default='gin', choices=['gin', 'gat'],
                        help='classifier backbone (default gin)')
    parser.add_argument('--n-heads', type=int, nargs='+', default=None,
                        help='attention heads per layer (scalar or per-layer list; default 4)')
    parser.add_argument('--head', default='linear', choices=['linear', 'ffn'],
                        help='head: linear = single Linear (default); ffn = two-layer MLP')
    parser.add_argument('--gnn-norm', default=None,
                        choices=['batchnorm', 'layernorm', 'none'],
                        help='inter-layer GNN normalization (default None = backbone '
                             'default, GIN: BatchNorm; the BatchNorm inside '
                             'GINELayer MLP is unaffected)')
    parser.add_argument('--readout', default='mean',
                        choices=['mean', 'meanmax', 'sum'],
                        help='graph readout pooling (gin only): mean = AvgPooling '
                             '(default); meanmax = concat(mean, max), robust to '
                             'subgraph scale; sum = SumPooling (scales with node '
                             'count)')
    parser.add_argument('--edge-encoder', default='linear',
                        choices=['linear', 'mlp'],
                        help='GNN+ edge feature integration: linear (default) / mlp')
    parser.add_argument('--ffn', action='store_true',
                        help='GNN+ per-layer FFN (residual + BN, pipeline epilogue)')
    parser.add_argument('--ffn-dropout', type=float, default=0.0,
                        help='FFN hidden dropout (applies when --ffn is on)')
    parser.add_argument('--pos-enc', default='none', choices=['none', 'rwse'],
                        help='GNN+ positional encoding: rwse (random-walk structural encoding)')
    parser.add_argument('--pos-enc-ksteps', type=int, default=8,
                        help='RWSE power steps K (with --pos-enc rwse; input dim = '
                             'node_feat_dim + K)')
    parser.add_argument('--init-checkpoint', default=None,
                        help='load weights from a classifier checkpoint to fine-tune '
                             '(structure from its config)')
    parser.add_argument('--resume', default='',
                        help='resume from a rolling checkpoint')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', default='',
                        help='experiment config JSON (explicit CLI args override)')
    parser.add_argument('--output-dir', default='',
                        help='default result/{dataset}/classifier')
    return parser


def _norm_heads(n_heads, n_layers: int):
    """Expand --n-heads scalar/single-item list per layer; None passes through."""
    if n_heads is None:
        return None
    if isinstance(n_heads, int):
        n_heads = [n_heads]
    if len(n_heads) == 1:
        n_heads = list(n_heads) * n_layers
    return list(n_heads)


def classifier_config_from_args(args, spec, init_cfg: dict):
    """Build a ClassifierConfig: init-checkpoint config first, then per-layer
    CLI values, then the uniform CLI form."""
    if init_cfg:
        cfg = ClassifierConfig.from_dict(init_cfg)
        cfg.edge_feat_dim = spec.edge_feat_dim
        cfg.node_feat_dim = spec.node_feat_dim
        cfg.n_classes = spec.n_classes
        return cfg
    if args.hidden_dims:
        dims = list(args.hidden_dims)
        drops = list(args.dropouts) if args.dropouts else [args.dropout] * len(dims)
        cfg = ClassifierConfig(
            backbone=args.backbone,
            edge_feat_dim=spec.edge_feat_dim, node_feat_dim=spec.node_feat_dim,
            n_classes=spec.n_classes, hidden_dims=dims, dropouts=drops,
            virtual_node=args.virtual_node, residual=args.residual,
            n_heads=_norm_heads(args.n_heads, len(dims)),
            head=args.head, gnn_norm=args.gnn_norm, readout=args.readout,
            edge_encoder=args.edge_encoder, ffn=args.ffn,
            ffn_dropout=args.ffn_dropout, pos_enc=args.pos_enc,
            pos_enc_ksteps=args.pos_enc_ksteps)
    else:
        cfg = ClassifierConfig.from_legacy(
            edge_feat_dim=spec.edge_feat_dim, n_classes=spec.n_classes,
            hidden_dim=args.hidden_dim, n_layers=args.n_layers,
            dropout=args.dropout, virtual_node=args.virtual_node,
            residual=args.residual, node_feat_dim=spec.node_feat_dim,
            backbone=args.backbone,
            n_heads=_norm_heads(args.n_heads, args.n_layers),
            head=args.head, gnn_norm=args.gnn_norm, readout=args.readout,
            edge_encoder=args.edge_encoder, ffn=args.ffn,
            ffn_dropout=args.ffn_dropout, pos_enc=args.pos_enc,
            pos_enc_ksteps=args.pos_enc_ksteps)
    return cfg


def train_one_epoch(clf, optimizer, graphs, labels, batch_size, device,
                    label_smoothing=0.0, class_weight=None):
    clf.train()
    total_loss = 0.0
    n_batches = 0
    correct = 0
    total = 0
    order = list(range(len(graphs)))
    random.shuffle(order)
    for start in tqdm(range(0, len(order), batch_size), desc="  Train"):
        idx = order[start:start + batch_size]
        batch_g = dgl.batch([graphs[i] for i in idx]).to(device)
        nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        y = labels[idx].to(device)
        logits = clf(batch_g, nids)
        loss = nn.functional.cross_entropy(logits, y, weight=class_weight,
                                           label_smoothing=label_smoothing)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += len(idx)
    return total_loss / max(n_batches, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(clf, graphs, labels, batch_size, device):
    clf.eval()
    correct = total = 0
    for start in range(0, len(graphs), batch_size):
        idx = list(range(start, min(start + batch_size, len(graphs))))
        batch_g = dgl.batch([graphs[i] for i in idx]).to(device)
        nids = torch.zeros(batch_g.num_nodes(), dtype=torch.long, device=device)
        logits = clf(batch_g, nids)
        pred = logits.argmax(dim=-1).cpu()
        correct += (pred == labels[idx]).sum().item()
        total += len(idx)
    return correct / max(total, 1)


def main():
    parser = build_parser()
    args = parser.parse_args()

    cfg_baseline = {}
    if args.config:
        cfg_baseline = load_json(args.config)
        for k, v in cfg_baseline.items():
            if hasattr(args, k) and getattr(args, k) == parser.get_default(k):
                setattr(args, k, v)
    args._defaults = {a.dest: parser.get_default(a.dest)
                      for a in parser._actions if a.dest != 'help'}
    from tools.config import require_args
    require_args(args, ['root'])

    set_all_seeds(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ds = get_dataset(args.dataset, root=args.root, max_graphs=args.max_graphs)
    spec = ds.spec
    print(f"Dataset {spec.name}: {spec.n_classes} classes, "
          f"node_feat_dim={spec.node_feat_dim}, edge_feat_dim={spec.edge_feat_dim}")

    train_graphs = ds.train_graphs
    test_graphs = ds.test_graphs
    train_labels = ds.train_labels
    test_labels = ds.test_labels
    print(f"Train graphs: {len(train_graphs)} (dataset has no validation split)")

    class_weight = compute_class_weight(args.class_weight, train_labels,
                                        spec.n_classes, device=device)

    init_ckpt = None
    init_cfg = {}
    if args.init_checkpoint:
        init_path = resolve_path(args.init_checkpoint)
        init_ckpt = torch.load(str(init_path), map_location=device, weights_only=False)
        init_cfg = init_ckpt.get('config', {}).get('classifier', init_ckpt.get('config', {}))
        if init_cfg.get('dataset') and init_cfg['dataset'] != spec.name:
            print(f"Warning: init checkpoint was trained on "
                  f"'{init_cfg['dataset']}', now '{spec.name}'; structure may not match")

    clf_cfg = classifier_config_from_args(args, spec, init_cfg)
    clf = GraphClassifier(config=clf_cfg).to(device)
    print(f"Params: {sum(p.numel() for p in clf.parameters()):,} | arch: {clf_cfg.to_dict()}")

    if init_ckpt is not None:
        clf.load_state_dict(init_ckpt['model_state_dict'], strict=True)
        print(f"Loaded init weights from {init_path} (fine-tune start)")

    optimizer = optim.Adam(clf.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler_patience > 0:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5,
            patience=args.scheduler_patience)

    out_dir = resolve_path(args.output_dir) if args.output_dir \
        else CODE_DIR / 'result' / spec.name / 'classifier'
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = f'{spec.name}_classifier'
    path = out_dir / f'{base_name}.pt'
    best_path = out_dir / f'best_{base_name}.pt'

    experiment_cfg = build_experiment_config(args, extra={
        'classifier': clf_cfg.to_dict(),
        'dataset_spec': {'name': spec.name, 'n_classes': spec.n_classes,
                         'node_feat_dim': spec.node_feat_dim,
                         'edge_feat_dim': spec.edge_feat_dim,
                         'needs_self_loop': spec.needs_self_loop,
                         'budget_options': list(spec.budget_options)},
        'device': str(device),
        'started_at': datetime.datetime.now().isoformat(timespec='seconds'),
    })
    save_json(experiment_cfg, out_dir / 'config.json')

    ckpt_config = {'classifier': clf_cfg.to_dict(), 'dataset': spec.name,
                   'n_classes': spec.n_classes}

    history = {'dataset': spec.name, 'epochs': []}
    best_acc = -1.0
    best_epoch = None
    start_epoch = 1

    if args.resume:
        resume_path = resolve_path(args.resume)
        ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
        clf.load_state_dict(ckpt['model_state_dict'], strict=True)
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        best_acc = ckpt.get('best_acc', -1.0)
        best_epoch = ckpt.get('best_epoch')
        hist_file = out_dir / 'history.json'
        if hist_file.exists():
            history = load_json(hist_file)
            history['epochs'] = [e for e in history['epochs']
                                 if e['epoch'] < start_epoch]
        print(f"Resumed from {resume_path} at epoch {start_epoch} "
              f"(best_acc={best_acc})")

    for epoch in range(start_epoch, args.epochs + 1):
        loss, train_acc = train_one_epoch(clf, optimizer, train_graphs,
                                          train_labels, args.batch_size, device,
                                          label_smoothing=args.label_smoothing,
                                          class_weight=class_weight)
        rec = {'epoch': epoch, 'loss': round(loss, 6),
               'train_acc': round(train_acc, 6), 'test_acc': None,
               'eval_checkpoint': None}
        ckpt = {'model_state_dict': clf.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch, 'best_acc': best_acc, 'best_epoch': best_epoch,
                'config': ckpt_config}
        torch.save(ckpt, str(path))
        line = f"Epoch {epoch}/{args.epochs}: loss={loss:.4f}  train_acc={train_acc:.4f}"

        stop = False
        if args.eval_every > 0 and epoch % args.eval_every == 0:
            test_acc = evaluate(clf, test_graphs, test_labels,
                                args.batch_size, device)
            rec['test_acc'] = round(test_acc, 6)
            eval_path = out_dir / f'eval_epoch{epoch}.pt'
            ckpt['train_acc'] = round(train_acc, 6)
            ckpt['test_acc'] = round(test_acc, 6)
            torch.save(ckpt, str(eval_path))
            rec['eval_checkpoint'] = eval_path.name
            improved = test_acc > best_acc
            if improved:
                best_acc, best_epoch = test_acc, epoch
                ckpt['best_acc'], ckpt['best_epoch'] = best_acc, best_epoch
                torch.save(ckpt, str(best_path))
            if scheduler is not None:
                scheduler.step(test_acc)
            line += (f"  test_acc={test_acc:.4f}  "
                     f"best={best_acc:.4f}(ep{best_epoch})")
            if (args.early_stop_patience > 0 and best_epoch is not None
                    and epoch - best_epoch >= args.early_stop_patience):
                stop = True
                line += "  [early stop]"
        history['epochs'].append(rec)
        history['best'] = (None if best_epoch is None
                           else {'epoch': best_epoch,
                                 'test_acc': round(best_acc, 6),
                                 'checkpoint': best_path.name})
        save_json(history, out_dir / 'history.json')
        print(line + f"  saved → {path}")
        if stop:
            break

    print(f"Done. Final checkpoint: {path}")
    if best_epoch is not None:
        print(f"Best: test_acc={best_acc:.4f} at epoch {best_epoch} → {best_path}")


if __name__ == '__main__':
    main()
