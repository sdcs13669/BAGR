"""REINFORCE fine-tuning for the evidence reader.

Starts from --bootstrap (any reader checkpoint; partial weight transfer across
architectures/datasets: copy same-shape, zero-pad last dim, skip others),
--from-scratch, or --resume. Reward judgment source --reward-source
(classifier / reader / both / soft-both) x binary/tanh/asymmetric transform;
batch advantage normalization with a negative-side down-weight
(--reward-neg-scale). Output: result/{dataset}/experiments/{exp}/evidence/rl/.
"""

import argparse
import datetime
import math
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))
sys.path = [p for p in sys.path
            if Path(p).resolve() != Path(__file__).resolve().parent]

import torch  # noqa: E402

from active_graph_reader import ReaderConfig, build_reader  # noqa: E402
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder  # noqa: E402
from datasets import get_dataset  # noqa: E402
from tools import cli as cli_utils  # noqa: E402
from tools.ckpt_io import load_optimizer, load_training_state, save_epoch_checkpoint, transfer_weights  # noqa: E402
from tools.config import build_experiment_config, load_json, save_json  # noqa: E402
from tools.paths import CODE_DIR  # noqa: E402
from tools.seeds import set_all_seeds  # noqa: E402
from train_reader.core import (bootstrap_reader, freeze_shared_layers,  # noqa: E402
                               load_classifier_frozen, load_reader_from_checkpoint,
                               rl_train_epoch)


def resolve(p) -> Path:
    from tools.paths import resolve as _resolve
    return _resolve(p)


def main():
    parser = argparse.ArgumentParser(description='REINFORCE fine-tuning (evidence reader)')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--root', default='', help='dataset root (required: CLI or --config)')
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='training graph limit (0 = all; use a small value for smoke tests)')
    parser.add_argument('--stratified', action='store_true',
                        help='stratify --max-graphs sampling by class (default: first N)')
    cli_utils.add_reader_args(parser, reader_kind_default='evidence',
                              temperature_default=0.2)
    parser.add_argument('--evidence-lambda', type=float, default=1.0,
                        help='weight of L_ev for the evidence reader (L = L_RL + lambda*L_ev)')
    parser.add_argument('--bootstrap', default='',
                        help='bootstrap reader checkpoint (one of --resume/--from-scratch)')
    parser.add_argument('--from-scratch', action='store_true',
                        help='initialize the reader from scratch and run RL directly '
                             '(structure from CLI reader args, n_classes from spec)')
    parser.add_argument('--classifier', default='', help='upper-bound classifier checkpoint (required: CLI or --config)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--graphs-per-batch', type=int, default=32)
    parser.add_argument('--seeds-per-graph', type=int, default=4)
    parser.add_argument('--reward', choices=['binary', 'tanh', 'asymmetric'],
                        default='binary')
    parser.add_argument('--reward-source',
                        choices=['classifier', 'reader', 'both', 'soft-both'],
                        default='classifier',
                        help='reward judgment source: classifier = frozen upper-bound '
                             'classifier on the revealed subgraph (default); '
                             'reader = the reader evidence-head terminal '
                             'judgment; both = both must be correct; soft-both = '
                             'R is the product of the two correct-class '
                             'probabilities (ignores --reward transform)')
    parser.add_argument('--reward-neg-scale', type=float, default=1,
                        help='down-weight factor for negative advantages after batch '
                             'normalization (1 = symmetric)')
    parser.add_argument('--no-adv-normalize', action='store_true',
                        help='disable batch advantage normalization (raw rewards as advantages)')
    parser.add_argument('--freeze', action='store_true',
                        help='freeze all but the feature projection layers (freeze ablation)')
    parser.add_argument('--edge-feat-dim', type=int, default=None,
                        help='override dataset edge feature dim')
    parser.add_argument('--node-feat-dim', type=int, default=None,
                        help='override dataset node feature dim')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--exp', default='default', help='experiment name')
    parser.add_argument('--output-dir', default='', help='override the default output directory')
    parser.add_argument('--resume', default='',
                        help='resume from an RL checkpoint; structure/hparams from its config')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', default='', help='experiment config JSON')
    args = parser.parse_args()

    cfg_baseline = load_json(args.config) if args.config else {}
    if cfg_baseline:
        for k, v in cfg_baseline.items():
            if hasattr(args, k) and getattr(args, k) == parser.get_default(k):
                setattr(args, k, v)
    cli_utils.attach_defaults(args, parser)
    from tools.config import require_args
    require_args(args, ['root']
                 + (['classifier'] if args.reward_source != 'reader' else []))

    if bool(args.bootstrap) + bool(args.resume) + bool(args.from_scratch) != 1:
        raise SystemExit("exactly one of --bootstrap / --resume / --from-scratch is required")

    set_all_seeds(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ds = get_dataset(args.dataset, root=args.root, splits=['train'],
                     max_graphs=args.max_graphs, stratified=args.stratified)
    spec = ds.spec
    budget_options = spec.budget_options
    ln_k = math.log(spec.n_classes)
    train_graphs = ds.train_graphs
    train_labels = ds.train_labels
    print(f"Train graphs: {len(train_graphs)} | budgets: {list(budget_options)}")

    classifier = None
    if args.reward_source != 'reader':
        clf_path = resolve(args.classifier)
        classifier = load_classifier_frozen(clf_path, device)
        print(f"Classifier loaded: {clf_path}")
        if args.reward_source == 'both':
            print("Reward source: both (classifier and evidence head must both be correct)")
        elif args.reward_source == 'soft-both':
            print("Reward source: soft-both (R = product of correct-class probabilities)")
    else:
        print("Reward source: reader own evidence-head judgment (no frozen classifier loaded)")

    start_epoch, training_log = 1, []
    resume_ckpt = None
    n_transferred, n_padded, skipped = 0, 0, []
    if args.resume:
        resume_path = resolve(args.resume)
        resume_ckpt = torch.load(str(resume_path), map_location='cpu',
                                 weights_only=False)
        reader = load_reader_from_checkpoint(resume_ckpt, device,
                                             stochastic=True,
                                             temperature=args.temperature)
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
    elif args.from_scratch:
        cfg_dict = cli_utils.reader_config_from_args(args)
        if cfg_dict.get('kind') == 'evidence':
            cfg_dict['n_classes'] = spec.n_classes
        reader_cfg = ReaderConfig.from_dict(cfg_dict)
        reader_cfg.edge_feat_dim = args.edge_feat_dim or spec.edge_feat_dim
        reader_cfg.node_feat_dim = args.node_feat_dim or spec.node_feat_dim
        reader_cfg.stochastic = True
        reader_cfg.validate()
        reader = build_reader(reader_cfg).to(device)
        print("From-scratch init: no BC/bootstrap starting point, direct RL")
        freeze_shared_layers(reader, args.freeze)
        optimizer = torch.optim.Adam(
            [p for p in reader.parameters() if p.requires_grad], lr=args.lr)
    else:
        bootstrap_path = resolve(args.bootstrap)
        src_ckpt = torch.load(str(bootstrap_path), map_location='cpu',
                              weights_only=False)
        reader, (n_transferred, n_padded, skipped) = bootstrap_reader(
            src_ckpt, device, stochastic=True, temperature=args.temperature)
        print(f"Bootstrap {args.bootstrap}: transferred={n_transferred}, "
              f"padded={n_padded}, skipped={len(skipped)} (new/mismatched layers)")
        freeze_shared_layers(reader, args.freeze)
        optimizer = torch.optim.Adam(
            [p for p in reader.parameters() if p.requires_grad], lr=args.lr)

    builder = SubgraphBuilder()
    if args.reward_source in ('reader', 'both', 'soft-both') \
            and not hasattr(reader, 'forward_states_evidence'):
        raise SystemExit("--reward-source reader/both/soft-both require the "
                         "evidence reader (own classification head)")
    total_params = sum(p.numel() for p in reader.parameters())
    trainable_params = sum(p.numel() for p in reader.parameters() if p.requires_grad)
    print(f"Reader: {reader.config.kind}, params={total_params:,} "
          f"(trainable {trainable_params:,})")

    output_dir = (resolve(args.output_dir) if args.output_dir else
                  CODE_DIR / 'result' / spec.name / 'experiments' / args.exp /
                  reader.config.kind / 'rl')
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and isinstance(resume_ckpt.get('config'), dict) \
            and resume_ckpt['config'].get('reader'):
        config = resume_ckpt['config']
        config['epochs'] = args.epochs
    else:
        config = build_experiment_config(args, extra={
            'reader': reader.config.to_dict(),
            'dataset': spec.name,
            'bootstrap': args.bootstrap,
            'classifier': str(clf_path) if classifier is not None else '',
            'reward_source': args.reward_source,
            'n_transferred': n_transferred,
            'n_padded': n_padded,
            'n_skipped': len(skipped),
            'freeze': args.freeze,
            'reward': args.reward,
            'reward_neg_scale': args.reward_neg_scale,
            'adv_normalize': not args.no_adv_normalize,
            'budget_options': list(budget_options),
            'epochs': args.epochs,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'started_at': datetime.datetime.now().isoformat(timespec='seconds'),
        })
    save_json(config, output_dir / 'config.json')
    print(f"Config saved: {output_dir / 'config.json'}")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        metrics = rl_train_epoch(
            reader, classifier, builder, optimizer,
            train_graphs, train_labels[:len(train_graphs)],
            args.graphs_per_batch, args.seeds_per_graph,
            device, args.reward, ln_k, budget_options,
            args.reward_neg_scale, evidence_lambda=args.evidence_lambda,
            adv_normalize=not args.no_adv_normalize,
            reward_source=args.reward_source)
        print(f"  Train: loss={metrics['loss']:.4f}  "
              f"avg_R={metrics['avg_R']:.3f}  acc={metrics['acc']:.3f}"
              + (f"  ev_acc={metrics['ev_acc']:.3f}"
                 if 'ev_acc' in metrics else '')
              + (f"  ev_acc_final={metrics['ev_acc_final']:.3f}"
                 if 'ev_acc_final' in metrics else ''))
        log_entry = {'epoch': epoch,
                     'train_loss': metrics['loss'],
                     'train_avg_R': metrics['avg_R'],
                     'train_acc': metrics['acc']}
        if 'ev_acc' in metrics:
            log_entry['ev_acc'] = metrics['ev_acc']
        if 'ev_acc_final' in metrics:
            log_entry['ev_acc_final'] = metrics['ev_acc_final']
        training_log.append(log_entry)
        save_epoch_checkpoint(
            output_dir / f'checkpoint_epoch{epoch}.pt',
            epoch=epoch, reader=reader, optimizer=optimizer,
            config=config, training_log=training_log)
        save_json(training_log, output_dir / 'training_log.json')
        print(f"  Saved: checkpoint_epoch{epoch}.pt")

    print(f"\nREINFORCE complete. Output: {output_dir}")


if __name__ == '__main__':
    main()
