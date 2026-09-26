"""Shared CLI argument groups and config-merge glue.

Argument groups shared by all training/generation scripts, plus attaching
argparse defaults (merge_args_with_config needs them to tell whether a CLI
flag was explicitly set).
"""

import argparse
from typing import Dict, Optional


def attach_defaults(args, parser: argparse.ArgumentParser) -> None:
    """Attach parser defaults to args._defaults (for tools.config.merge)."""
    args._defaults = {a.dest: parser.get_default(a.dest)
                      for a in parser._actions if a.dest != 'help'}


def add_dataset_args(parser: argparse.ArgumentParser,
                     default_dataset: str = 'cifar10') -> None:
    parser.add_argument('--dataset', default=default_dataset,
                        help='dataset registry name')
    parser.add_argument('--root', required=True,
                        help='dataset root directory')
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='max graphs per split (0 = no limit)')
    parser.add_argument('--device', default='cuda:0')


def add_reader_args(parser: argparse.ArgumentParser,
                    reader_kind_default: str = 'evidence',
                    temperature_default: float = 1.0) -> None:
    """Reader structure args (keys match ReaderConfig.to_dict)."""
    parser.add_argument('--reader-kind', choices=['evidence'],
                        default=reader_kind_default)
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='uniform hidden width; mutually exclusive with --hidden-dims')
    parser.add_argument('--n-layers', type=int, default=2,
                        help='encoder GNN layers (uniform form)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout (uniform form)')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=None,
                        help='per-layer encoder widths (overrides --hidden-dim/--n-layers)')
    parser.add_argument('--encoder-dropouts', type=float, nargs='+', default=None,
                        help='per-layer encoder dropouts')
    parser.add_argument('--scorer-hidden-dims', type=int, nargs='+', default=None,
                        help='scorer MLP per-layer widths')
    parser.add_argument('--temperature', type=float, default=temperature_default,
                        help='sampling temperature (RL/eval; BC uses 1.0)')


def reader_config_from_args(args) -> Dict:
    """CLI reader args -> ReaderConfig-compatible dict."""
    cfg: Dict[str, object] = {
        'kind': getattr(args, 'reader_kind', 'evidence'),
        'edge_feat_dim': getattr(args, 'edge_feat_dim', 7),
        'node_feat_dim': getattr(args, 'node_feat_dim', 0),
        'temperature': getattr(args, 'temperature', 1.0),
    }
    hidden_dims = getattr(args, 'hidden_dims', None)
    dropouts = getattr(args, 'encoder_dropouts', None)
    if hidden_dims:
        cfg['hidden_dims'] = list(hidden_dims)
        cfg['dropouts'] = list(dropouts) if dropouts else \
            [getattr(args, 'dropout', 0.1)] * len(hidden_dims)
    else:
        cfg['hidden_dim'] = getattr(args, 'hidden_dim', 128)
        cfg['n_hidden_dims'] = getattr(args, 'n_layers', 2)
        cfg['dropout'] = getattr(args, 'dropout', 0.1)
    scorer = getattr(args, 'scorer_hidden_dims', None)
    if scorer:
        cfg['scorer_hidden_dims'] = list(scorer)
    if cfg['kind'] == 'evidence':
        cfg['n_classes'] = getattr(args, 'n_classes', 0)
    return cfg
