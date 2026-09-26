"""Soft-Mask scorer training: standard supervision on full graphs (single-graph
forward; --batch-size controls graphs per optimizer step).

The checkpoint feeds SoftmaskStrategy for protocol-compliant evaluation.
Hyperparameters follow the original defaults (lr 5e-4 / wd 1e-4 / seed 777;
hidden 128 to match run_sagpool). Batching uses gradient accumulation; the
model has no BatchNorm so accumulated mean gradients are mathematically
equal to a true batch; batch 1 degrades to the plain loop. Self-loops are
normalized (remove+add) to match the strategy graph contract."""

import argparse
import datetime
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CODE_DIR = _REPO_ROOT / 'code'
for _p in (_CODE_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402
import dgl  # noqa: E402
from tqdm import tqdm  # noqa: E402

from baseline.softmask import SoftMaskModel  # noqa: E402
from datasets import get_dataset  # noqa: E402
from tools.config import save_json  # noqa: E402
from tools.paths import REPO_ROOT  # noqa: E402
from tools.seeds import set_all_seeds  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Soft-Mask scorer training (full graph)')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--root', required=True, help='dataset root')
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='max graphs per split (0 = all)')
    parser.add_argument('--stratified', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='stratified max_graphs truncation by class with a fixed seed '
                             '(per split); --no-stratified falls back to prefix truncation')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=1,
                        help='graphs per optimizer step (>1 uses gradient accumulation)')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=777, help='matches the original default')
    parser.add_argument('--output', default='',
                        help='checkpoint output path (default baseline/artifacts/{ds}/softmask.pt)')
    return parser


@torch.no_grad()
def evaluate(model, graphs, labels, device):
    model.eval()
    correct = 0
    for i in range(len(graphs)):
        logits = model(graphs[i].to(device))
        correct += int(logits.argmax(dim=-1).item() == int(labels[i].item()))
    model.train()
    return correct / max(len(graphs), 1)


def main():
    args = build_parser().parse_args()
    set_all_seeds(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ds = get_dataset(args.dataset, root=args.root,
                     splits=['train', 'test'], max_graphs=args.max_graphs,
                     stratified=args.stratified)
    spec = ds.spec
    print(f"Dataset {spec.name}: train={len(ds.train_graphs)} test={len(ds.test_graphs)}")

    train_graphs = [dgl.add_self_loop(dgl.remove_self_loop(g))
                    for g in tqdm(ds.train_graphs, desc='Self-loop train')]
    test_graphs = [dgl.add_self_loop(dgl.remove_self_loop(g))
                   for g in tqdm(ds.test_graphs, desc='Self-loop test')]
    train_labels = ds.train_labels
    test_labels = ds.test_labels

    model = SoftMaskModel(
        node_feat_dim=spec.node_feat_dim, edge_feat_dim=spec.edge_feat_dim,
        n_classes=spec.n_classes, hidden=args.hidden, n_layers=args.layers,
        dropout=args.dropout).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    order = list(range(len(train_graphs)))
    best_acc, best_state = -1.0, None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(order)
        correct = total = 0
        chunks = [order[i:i + args.batch_size]
                  for i in range(0, len(order), args.batch_size)]
        for chunk in tqdm(chunks, desc=f'Epoch {epoch}/{args.epochs}'):
            optimizer.zero_grad()
            chunk_correct = 0
            for i in chunk:
                g = train_graphs[i].to(device)
                y = torch.tensor([int(train_labels[i].item())], device=device)
                logits = model(g)
                (torch.nn.functional.cross_entropy(logits, y) / len(chunk)).backward()
                chunk_correct += int(logits.argmax(dim=-1).item() == y.item())
            optimizer.step()
            correct += chunk_correct
            total += len(chunk)
        test_acc = evaluate(model, test_graphs, test_labels, device)
        history.append({'epoch': epoch, 'train_acc': round(correct / max(total, 1), 6),
                        'test_acc': round(test_acc, 6)})
        print(f"Epoch {epoch}: train_acc={correct / max(total, 1):.4f} "
              f"test_acc={test_acc:.4f}")
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    out = Path(args.output) if args.output else None
    out_path = out if (out is not None and out.is_absolute()) else (
        out if out is not None
        else REPO_ROOT / 'baseline' / 'artifacts' / spec.name / 'softmask.pt')
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': best_state if best_state is not None
        else model.state_dict(),
        'config': {'dataset': spec.name, 'node_feat_dim': spec.node_feat_dim,
                   'edge_feat_dim': spec.edge_feat_dim,
                   'n_classes': spec.n_classes, 'hidden': args.hidden,
                   'n_layers': args.layers, 'dropout': args.dropout,
                   'batch_size': args.batch_size},
        'test_acc': best_acc, 'history': history,
        'trained_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'max_graphs': args.max_graphs, 'epochs': args.epochs,
    }, str(out_path))
    save_json({'dataset': spec.name, 'history': history, 'test_acc': best_acc,
               'config': {'hidden': args.hidden, 'n_layers': args.layers,
                          'dropout': args.dropout,
                          'batch_size': args.batch_size,
                          'max_graphs': args.max_graphs,
                          'stratified': args.stratified,
                          'epochs': args.epochs}},
              out_path.with_suffix('.history.json'))
    print(f"Best test_acc={best_acc:.4f} → {out_path}")


if __name__ == '__main__':
    main()
