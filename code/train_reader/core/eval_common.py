"""Shared pipeline for the two eval entry points (eval_reader / eval_baselines).

- add_common_args: CLI args shared by both evaluation entry points;
- run_split_eval: per-split load-and-evaluate (load one split, evaluate,
  free memory, then the next; the JSON is incrementally written after each
  split so a later crash does not lose finished results).

The eval_split callback returns Iterable[(strategy_key, results)]; keys land
in the JSON {split: {strategy_key: {...}}} subtree (reader uses "learnable").
The config dict must contain n_graphs and full_graph_ref (accumulated per
split); other fields are up to each entry point.
"""

import argparse
import gc

import torch

from datasets import get_dataset, release_dataset
from tools.config import save_json
from .eval_loop import build_eval_json, evaluate_classifier_full, merge_eval_json


def resolve(p):
    from tools.paths import resolve as _resolve
    return _resolve(p)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """CLI args shared by both eval scripts (--checkpoint/--strategy added separately)."""
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--root', required=True, help='dataset root')
    parser.add_argument('--classifier', required=True, help='upper-bound classifier checkpoint')
    parser.add_argument('--splits', nargs='+', default=['valid', 'test'],
                        choices=['train', 'valid', 'test'],
                        help='splits to evaluate (missing splits are skipped with a warning)')
    parser.add_argument('--max-graphs', type=int, default=0,
                        help='evaluation graph limit per split (0 = all)')
    parser.add_argument('--stratified', action='store_true',
                        help='stratify --max-graphs sampling by class (same seed/selection as '
                             'training so subsets match)')
    parser.add_argument('--argmax', action=argparse.BooleanOptionalAction, default=True,
                        help='argmax selection (default on; --no-argmax samples from the policy)')
    parser.add_argument('--n-seeds', type=int, default=4,
                        help='deterministic seeds per graph (default 4, averaged)')
    parser.add_argument('--ref-batch-size', type=int, default=256,
                        help='batch size for the full-graph reference evaluation (reduce on '
                             'OOM with large-graph datasets, e.g. 32)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', default='',
                        help='output directory (default: reader -> checkpoint directory;'
                             'baseline → result/{ds}/fixed）')


_CALIBRATION_NOTE = ('[[max_prob (4-dp), correct], '
                     '...] per (graph, seed) sample')


def run_split_eval(args, classifier, budget_options, config, out_path,
                   eval_split, calibration=None):
    """Per-split load-and-evaluate with incremental JSON writes.




    Each split: load graphs -> full-graph classifier reference ->
    per-strategy eval_split callback -> merge JSON -> free the split data
    (cache eviction + gc + empty_cache). When calibration is set, also write
    out_path.parent/calibration.json per split.
    """
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    parts = []
    evaluated = []

    def _warn_conflict(path):
        print(f"[warn] merge conflict in {path} (later part wins)")

    for split in list(dict.fromkeys(args.splits)):
        ds = get_dataset(args.dataset, root=args.root, splits=(split,),
                         max_graphs=args.max_graphs, stratified=args.stratified)
        graphs = getattr(ds, f'{split}_graphs')
        labels = getattr(ds, f'{split}_labels')
        if len(graphs) == 0:
            print(f"[skip] dataset '{args.dataset}' has no {split} split (or empty)")
            release_dataset(ds)
            continue

        ref = evaluate_classifier_full(classifier, graphs, labels, device,
                                       batch_size=args.ref_batch_size)
        config['full_graph_ref'][split] = round(ref, 6)
        config['n_graphs'][split] = len(graphs)
        print(f"[{split}] loaded {len(graphs)} graphs; "
              f"full-graph reference acc = {ref:.4f}")

        for key, results in eval_split(split, graphs, labels):
            parts.append(build_eval_json(results, split, key,
                                         args.n_seeds, budget_options, config={}))
            print(f"[{split}] {key}: "
                  + '  '.join(f"B{int(b * 100)}%={results[f'acc_{int(b * 100)}%']:.4f}"
                              for b in budget_options)
                  + f"  overall={results['acc_overall']:.4f}")
            if 'acc_clf_overall' in results:
                print(f"[{split}] {key} [ref]: "
                      + '  '.join(f"B{int(b * 100)}%={results[f'acc_clf_{int(b * 100)}%']:.4f}"
                                  for b in budget_options)
                      + f"  overall={results['acc_clf_overall']:.4f}"
                      + "(evidence terminal = own evidence head; ref = frozen classifier)")

        evaluated.append(split)
        json_out = merge_eval_json(parts, on_conflict=_warn_conflict)
        json_out['config'] = config
        save_json(json_out, out_path)
        print(f"[{split}] JSON saved to {out_path}")
        if calibration:
            save_json({'calibration': calibration,
                       'config': {'note': _CALIBRATION_NOTE}},
                      out_path.parent / 'calibration.json')

        del graphs, labels
        release_dataset(ds)
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if not evaluated:
        raise SystemExit(f"all requested splits {args.splits} are empty for dataset '{args.dataset}'")
