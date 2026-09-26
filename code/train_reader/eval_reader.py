"""Evaluation for the trained evidence reader.

Reader structure is rebuilt entirely from the checkpoint embedded config
(no model CLI args). One run evaluates all requested splits (default
valid+test; missing splits skipped) and merges results into a single
unified-schema JSON -> <checkpoint dir>/eval_learnable.json (--output-dir to
override).

The evidence reader rolls out via forward_states_evidence (selection identical
to the base pipeline) and its terminal judgment is the argmax of ev_logits
from one extra evidence forward after the budget is exhausted; the frozen
upper-bound classifier on the same final subgraphs is recorded as the
acc_clf_* reference keys.

--export-confidence writes per-sample [max class confidence, correct] to
calibration.json next to the checkpoint (ev = evidence-head judgment,
clf = frozen classifier).
"""

import argparse
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
from tools.seeds import SEED_METHOD_STR  # noqa: E402
from train_reader.core import (add_common_args, eval_resolve,  # noqa: E402
                               load_classifier_frozen,
                               load_reader_from_checkpoint, run_split_eval)
from train_reader.core.eval_loop import evaluate_reader  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='evidence reader evaluation')
    add_common_args(parser)
    parser.add_argument('--checkpoint', required=True,
                        help='reader checkpoint (structure rebuilt from embedded config)')
    parser.add_argument('--chunk-size', type=int, default=32,
                        help='graph batch size for cross-graph lockstep rollout')
    parser.add_argument('--export-confidence', action='store_true',
                        help='export per-sample [max class confidence, correct] to '
                             'calibration.json next to the checkpoint '
                             '(ev = evidence head / clf = frozen classifier)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    spec = get_dataset(args.dataset, root=args.root, splits=(),
                       max_graphs=args.max_graphs,
                       stratified=args.stratified).spec
    budget_options = spec.budget_options

    classifier = load_classifier_frozen(eval_resolve(args.classifier), device)
    print(f"Classifier: {args.classifier}")
    builder = SubgraphBuilder()

    ckpt_path = eval_resolve(args.checkpoint)
    print(f"Checkpoint: {ckpt_path}")
    reader = load_reader_from_checkpoint(ckpt_path, device,
                                         stochastic=False, temperature=1.0)
    print(f"Reader: kind={reader.config.kind} "
          f"hidden={reader.config.encoder_hidden_dims} "
          f"(from checkpoint config)")

    out_dir = (eval_resolve(args.output_dir) if args.output_dir
               else ckpt_path.parent)
    out_path = out_dir / 'eval_learnable.json'
    out_dir.mkdir(parents=True, exist_ok=True)

    calibration = {} if args.export_confidence else None
    config = {
        'dataset': spec.name,
        'mode': 'reader',
        'strategies': ['learnable'],
        'strategy_or_checkpoint': str(ckpt_path),
        'checkpoint': str(ckpt_path),
        'classifier': str(eval_resolve(args.classifier)),
        'budgets': list(budget_options),
        'n_graphs': {},
        'n_seeds': args.n_seeds,
        'argmax': args.argmax,
        'seed_method': SEED_METHOD_STR,
        'full_graph_ref': {},
    }

    def eval_split(split, graphs, labels):
        results = evaluate_reader(reader, classifier, builder, graphs, labels,
                                  budget_options, device, args.argmax,
                                  chunk_size=args.chunk_size,
                                  n_seeds=args.n_seeds,
                                  confidence_out=(calibration.setdefault(split, {})
                                                  if calibration is not None
                                                  else None))
        yield 'learnable', results

    run_split_eval(args, classifier, budget_options, config, out_path,
                   eval_split, calibration=calibration)
    print(f"JSON saved to {out_path}")
    if calibration:
        calib_path = out_dir / 'calibration.json'
        print(f"Calibration saved to {calib_path}")
        print(f"plot: python comparison/plot_calibration.py --json {calib_path}")
    print(f"plot: python comparison/plot_eval_results.py --json {out_path}")


if __name__ == '__main__':
    main()
