"""Evaluation for fixed strategies and protocol-adapted baselines.

Covers fixed strategies (random / bfs / max-frontier-degree, in
reader/fixed/) and the adapted baselines registered by baseline/strategies.py
(random-walk / sagpool / softmask / agentnet; the latter three need a scorer
checkpoint via --strategy-ckpt, one per command).

One run evaluates all requested splits (default valid+test; splits missing
from the dataset are skipped) and merges results into a single unified-schema
JSON -> result/{ds}/fixed/eval_fixed.json (override with --output-dir).

Protocol: budget = number of revealed nodes excluding the seed, tiers from
DatasetSpec (default 0.05/0.10/0.15); deterministic seed formula (tools/seeds);
frontier edge features are visible before reveal at zero budget; frontier node
features are masked; random-type baselines seed their RNG by (seed_node,
n_nodes) so every trajectory is reproducible; terminal judgment by the frozen
upper-bound classifier, with agentnet own-head results additionally recorded
as acc_own_* (deterministic argmax walk).
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
from active_graph_reader.reader.fixed import create_fixed_reader  # noqa: E402
from datasets import get_dataset  # noqa: E402
from tools.paths import CODE_DIR  # noqa: E402
from tools.seeds import SEED_METHOD_STR  # noqa: E402
from train_reader.core import (add_common_args, eval_resolve,  # noqa: E402
                               evaluate_fixed, load_classifier_frozen,
                               run_split_eval)


def main():
    parser = argparse.ArgumentParser(description='fixed-strategy / baseline evaluation')
    add_common_args(parser)
    parser.add_argument('--strategy', nargs='+', required=True,
                        choices=['random', 'bfs', 'max-frontier-degree',
                                 'random-walk', 'sagpool', 'subgraphx',
                                 'softmask', 'agentnet'],
                        help='fixed strategy names (multiple allowed); random-walk/sagpool/'
                             'softmask/agentnet are protocol-adapted baselines; '
                             'sagpool/softmask/agentnet need a scorer checkpoint '
                             '(--strategy-ckpt, one per command)')
    parser.add_argument('--strategy-ckpt', default='',
                        help='sagpool scorer checkpoint'
                             ' (default baseline/artifacts/{ds}/sagpool.pt)')
    parser.add_argument('--judge', default='clf', choices=['clf', 'own'],
                        help='terminal judgment: clf = frozen upper-bound classifier '
                             '(default); own = the strategy own head (only '
                             'sagpool/softmask/agentnet; the frozen classifier is '
                             'then recorded as the acc_clf_* reference)')
    parser.add_argument('--search-rollouts', type=int, default=20,
                        help='subgraphx MCTS rollout count (official default 20)')
    parser.add_argument('--search-expand', type=int, default=12,
                        help='subgraphx max child nodes per expansion (official expand_atoms=12)')
    parser.add_argument('--search-max-evals', type=int, default=2048,
                        help='subgraphx max unique coalitions evaluated per planning step')
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

    print(f"Strategies: {' '.join(args.strategy)}")
    if args.judge == 'own':
        own_capable = {'sagpool', 'softmask', 'agentnet'}
        bad = [s for s in args.strategy if s not in own_capable]
        if bad:
            raise SystemExit(f"--judge own requires own-head strategies, got: {' '.join(bad)}"
                             " (choose sagpool/softmask/agentnet)")
    if any(s in ('random-walk', 'sagpool', 'subgraphx',
                 'softmask', 'agentnet') for s in args.strategy):
        import sys as _sys
        from tools.paths import REPO_ROOT
        if str(REPO_ROOT) not in _sys.path:
            _sys.path.insert(0, str(REPO_ROOT))
        from baseline.strategies import register_baseline_strategies
        register_baseline_strategies()

    agents = []
    for strategy in args.strategy:
        if strategy == 'sagpool':
            from baseline.sagpool import SagpoolStrategy
            ckpt = (eval_resolve(args.strategy_ckpt) if args.strategy_ckpt
                    else CODE_DIR.parent / 'baseline' / 'artifacts'
                         / spec.name / 'sagpool.pt')
            if not ckpt.exists():
                raise SystemExit(
                    f"sagpool scorer checkpoint not found: {ckpt}\n"
                    f"train it first: python baseline/run_sagpool.py "
                    f"--dataset {spec.name} --root <root>")
            agent = SagpoolStrategy()
            agent.load(str(ckpt), device)
            print(f"SAGPool scorer: {ckpt}")
        elif strategy in ('softmask', 'agentnet'):
            if strategy == 'softmask':
                from baseline.softmask import SoftmaskStrategy
                agent = SoftmaskStrategy()
            else:
                from baseline.agentnet import AgentnetStrategy
                agent = AgentnetStrategy()
            ckpt = (eval_resolve(args.strategy_ckpt) if args.strategy_ckpt
                    else CODE_DIR.parent / 'baseline' / 'artifacts'
                         / spec.name / f'{strategy}.pt')
            if not ckpt.exists():
                raise SystemExit(
                    f"{strategy} scorer checkpoint not found: {ckpt}\n"
                    f"train it first: python baseline/run_{strategy}.py "
                    f"--dataset {spec.name} --root <root>")
            agent.load(str(ckpt), device)
            print(f"{strategy} scorer: {ckpt}")
        elif strategy == 'subgraphx':
            from baseline.subgraphx import SubgraphXStrategy
            agent = SubgraphXStrategy(
                n_rollout=args.search_rollouts,
                expand_atoms=args.search_expand,
                max_evals=args.search_max_evals)
            agent.configure(classifier, device)
            print(f"SubgraphX planner: rollouts={args.search_rollouts} "
                  f"expand={args.search_expand} "
                  f"max_evals={args.search_max_evals} (planner budget)")
        else:
            agent = create_fixed_reader(strategy)
        agents.append(agent)

    out_dir = (eval_resolve(args.output_dir) if args.output_dir
               else CODE_DIR / 'result' / spec.name / 'fixed')
    out_path = out_dir / 'eval_fixed.json'
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        'dataset': spec.name,
        'mode': 'fixed',
        'strategies': list(args.strategy),
        'strategy_or_checkpoint': ' '.join(args.strategy),
        'classifier': str(eval_resolve(args.classifier)),
        'budgets': list(budget_options),
        'n_graphs': {},
        'n_seeds': args.n_seeds,
        'argmax': args.argmax,
        'seed_method': SEED_METHOD_STR,
        'judge': args.judge,
        'full_graph_ref': {},
    }

    def eval_split(split, graphs, labels):
        for strategy, agent in zip(args.strategy, agents):
            results = evaluate_fixed(None, classifier, builder, graphs, labels,
                                     budget_options, device, args.argmax,
                                     args.n_seeds, agent=agent,
                                     judge=args.judge)
            yield strategy, results

    run_split_eval(args, classifier, budget_options, config, out_path,
                   eval_split)
    print(f"JSON saved to {out_path}")
    print(f"plot: python comparison/plot_eval_results.py --json {out_path}")


if __name__ == '__main__':
    main()
