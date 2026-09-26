"""Enrich and balance trajectories: exactly per-class-target per budget tier
and class.

Resume support: each iteration writes validated new trajectories to
--state-file (default <output-dir>/balance_state.pt); the next run reloads it
and skips finished classes. --labels selects the classes to balance;
--no-generate truncates only; --deficit-only generates only for deficit
tiers."""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))
sys.path = [p for p in sys.path
            if Path(p).resolve() != Path(__file__).resolve().parent]

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from active_graph_reader.engine.graph_state import GraphState  # noqa: E402
from active_graph_reader.gnn.subgraph_builder import SubgraphBuilder  # noqa: E402
from datasets import get_dataset  # noqa: E402
from tools.paths import CODE_DIR  # noqa: E402
from tools.seeds import set_all_seeds  # noqa: E402
from tools.trajectory_io import load_trajectory_file, ratio_key  # noqa: E402
from trajectories.core.filtering import dedup_group, select_top_k  # noqa: E402
from trajectories.core.rollout import (build_classifier, build_reader,  # noqa: E402
                                       build_edge_key_map, rollout_batch,
                                       validate_candidates)
from trajectories.core.trajectory import Trajectory  # noqa: E402


def resolve(p: str) -> Path:
    from tools.paths import resolve as _resolve
    return _resolve(p)


def count_after_filter(pool, train_labels, sim_threshold, top_k):
    """Per-class counts after simulated dedup + per-graph top-k.

    Uses the same filtering as the final balancing stage so pool counts do
    not overstate post-filter counts.
    """
    by_graph: dict = defaultdict(list)
    for t in pool:
        by_graph[t.graph_idx].append(t)
    cnt = defaultdict(int)
    for trajs in by_graph.values():
        recs = sorted(((-t.ce, t.teacher_type, t) for t in trajs),
                      key=lambda r: r[0])
        ded = dedup_group(recs, sim_threshold)
        for traj in select_top_k(ded, top_k, require_diversity=False):
            cnt[int(train_labels[traj.graph_idx])] += 1
    return cnt


def main():
    parser = argparse.ArgumentParser(description='trajectory balancing (per tier per class target)')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--root', required=True,
                        help='dataset root (e.g. D:/data/superpixels)')
    parser.add_argument('--reader', required=True,
                        help='reader checkpoint (BC product)')
    parser.add_argument('--classifier', required=True,
                        help='upper-bound classifier checkpoint')
    parser.add_argument('--budgets', type=float, nargs='+', default=None,
                        help='budget ratios (default spec.budget_options)')
    parser.add_argument('--input-dir', default='',
                        help='existing filtered trajectory directory')
    parser.add_argument('--output-dir', default='',
                        help='output directory (default data/{dataset}/trajectories_balanced)')
    parser.add_argument('--per-class-target', type=int, default=2000,
                        help='target trajectories per class per tier')
    parser.add_argument('--labels', type=int, nargs='+', default=None,
                        help='classes to balance this run (default all; other classes only get '
                             'dedup + per-graph top-k)')
    parser.add_argument('--state-file', default=None,
                        help='progress state file (default <output-dir>/balance_state.pt)')
    parser.add_argument('--sim-threshold', type=float, default=0.8,
                        help='cosine similarity threshold for within-graph dedup')
    parser.add_argument('--top-k', type=int, default=4,
                        help='max trajectories kept per graph after dedup')
    parser.add_argument('--attempts', type=int, default=20,
                        help='seeds tried per graph per tier (k)')
    parser.add_argument('--graphs-per-round', type=int, default=1000,
                        help='training graphs sampled per class per round')
    parser.add_argument('--deficit-only', action='store_true',
                        help='generate only for deficit budget tiers')
    parser.add_argument('--max-rounds', type=int, default=10,
                        help='epoch cap (prevents non-convergence)')
    parser.add_argument('--no-generate', action='store_true',
                        help='skip generation; only dedup -> top-k -> per-class truncation')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_all_seeds(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ds = get_dataset(args.dataset, root=args.root or None, splits=['train'])
    spec = ds.spec
    train_graphs, train_labels = ds.train_graphs, ds.train_labels
    budgets = list(args.budgets or spec.budget_options)
    n_classes = spec.n_classes
    print(f"Dataset: {spec.name} | train={len(train_graphs)} "
          f"| classes={n_classes} | budgets={budgets}")

    r_path = resolve(args.reader)
    reader = build_reader(torch.load(str(r_path), map_location=device,
                                     weights_only=False), device)
    c_path = resolve(args.classifier)
    classifier = build_classifier(torch.load(str(c_path), map_location=device,
                                             weights_only=False), device)
    builder = SubgraphBuilder()
    print(f"Reader: {r_path} | Classifier: {c_path}")

    by_label = defaultdict(list)
    for i in range(len(train_graphs)):
        by_label[int(train_labels[i])].append(i)

    targets = list(args.labels) if args.labels else list(range(n_classes))
    print(f"Target labels: {targets}")

    input_dir = resolve(args.input_dir) if args.input_dir else \
        CODE_DIR / 'data' / spec.name / 'trajectories_filtered'
    output_dir = resolve(args.output_dir) if args.output_dir else \
        CODE_DIR / 'data' / spec.name / 'trajectories_balanced'
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state_file) if args.state_file else (
        output_dir / 'balance_state.pt')
    if not state_path.is_absolute():
        state_path = CODE_DIR / state_path

    pools = {}
    for ratio in budgets:
        key = ratio_key(ratio)
        fpath = input_dir / f'trajectories_r{key}.pt'
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found (no existing trajectories, all generated)")
            pools[key] = []
            continue
        data = load_trajectory_file(fpath)
        pool = []
        for tname, per_graph in data['trajectories'].items():
            for gi, lst in enumerate(per_graph):
                pool.extend(lst)
        pools[key] = pool
        print(f"  {key}: existing pool = {len(pool)}")

    if state_path.exists():
        state = torch.load(str(state_path), map_location='cpu',
                           weights_only=False)
        state_gen = state.get('generated', {})
        n_loaded = 0
        for key, trajs in state_gen.items():
            pools.setdefault(key, []).extend(trajs)
            n_loaded += len(trajs)
        print(f"  resume: loaded {n_loaded} generated trajectories from "
              f"{state_path}")
        rng = random.Random(args.seed + n_loaded)
    else:
        state_gen = {}

    report = {'per_class_target': args.per_class_target, 'rounds': {}}
    for round_i in range(1, args.max_rounds + 1):
        print(f"\n===== Round {round_i} =====")
        cur_counts = {}
        for ratio in budgets:
            key = ratio_key(ratio)
            cur_counts[key] = count_after_filter(
                pools[key], train_labels, args.sim_threshold, args.top_k)

        _all_done = True
        for ratio in budgets:
            key = ratio_key(ratio)
            per_label = cur_counts[key]
            print(f"  {key}: " + ", ".join(
                f"l{i}={per_label.get(i, 0)}" for i in range(n_classes))
                + f"  (target={args.per_class_target})")
            for lab in targets:
                if per_label.get(lab, 0) < args.per_class_target:
                    _all_done = False
        report['rounds'][round_i] = {
            key: dict(cur_counts[key]) for key in cur_counts}
        if _all_done:
            print("=== all target labels reach target ===")
            break

        if args.no_generate:
            print("=== --no-generate: no new trajectories, straight to final balancing ===")
            break

        n_cand = n_valid = 0
        for lab in targets:
            ok_everywhere = all(
                cur_counts[ratio_key(r)].get(lab, 0) >= args.per_class_target
                for r in budgets)
            if ok_everywhere:
                continue
            gen_ratios = ([r for r in budgets
                           if cur_counts[ratio_key(r)].get(lab, 0)
                           < args.per_class_target]
                          if args.deficit_only else budgets)
            pool_g = by_label[lab]
            n_sample = min(args.graphs_per_round, len(pool_g))
            sampled = rng.sample(pool_g, n_sample)
            desc = f"  gen label {lab}"
            if args.deficit_only:
                desc += " [" + ",".join(ratio_key(r) for r in gen_ratios) + "]"
            for gi in tqdm(sampled, desc=desc,
                           unit='graphs', leave=False):
                g = train_graphs[gi]
                n_nodes = g.num_nodes()
                edge_map = build_edge_key_map(g)
                g_states = []  # (ratio, state)
                for _try in range(args.attempts):
                    seed_node = rng.randint(0, n_nodes - 1)
                    for ratio in gen_ratios:
                        B = max(1, int(ratio * n_nodes))
                        state = GraphState(g, seed_node=seed_node,
                                           edge_key_to_id=edge_map)
                        state._budget_limit = B
                        g_states.append((ratio, state))
                results = rollout_batch(reader, [s for _r, s in g_states], rng)
                cands = []  # (graph_idx, ratio, traj, state)
                for (ratio, state), (reveal_order, fin_state) in zip(
                        g_states, results):
                    if not reveal_order:
                        continue
                    traj = Trajectory(
                        graph_idx=gi,
                        seed_node=state.seed_node,
                        reveal_ratio=ratio,
                        reveal_order=reveal_order,
                        final_pred=-1,
                        is_correct=True,
                        teacher_type='gradient',
                    )
                    cands.append((gi, ratio, traj, fin_state))
                n_cand += len(cands)
                for gi2, ratio2, traj2 in validate_candidates(
                        cands, builder, classifier, device, train_labels,
                        args.batch_size):
                    pools[ratio_key(ratio2)].append(traj2)
                    state_gen.setdefault(ratio_key(ratio2), []).append(traj2)
                    n_valid += 1
        print(f"  generated {n_cand}, validated {n_valid}")
        torch.save({'generated': state_gen,
                    'config': {'target_labels': targets,
                               'per_class_target': args.per_class_target}},
                   str(state_path))

    balance_report = {}
    for ratio in budgets:
        key = ratio_key(ratio)
        pool = pools[key]
        by_graph: dict = defaultdict(list)
        for t in pool:
            by_graph[t.graph_idx].append(t)
        kept_by_graph = []
        n_dedup = n_after_dedup = 0
        for g_idx, trajs in by_graph.items():
            recs = sorted(
                ((-t.ce, t.teacher_type, t) for t in trajs),
                key=lambda r: r[0])
            ded = dedup_group(recs, args.sim_threshold)
            n_dedup += len(recs) - len(ded)
            n_after_dedup += len(ded)
            kept_by_graph.extend(select_top_k(
                ded, args.top_k, require_diversity=False))

        by_class: dict = defaultdict(list)
        for t in kept_by_graph:
            by_class[int(train_labels[t.graph_idx])].append(t)
        kept_final, per_class = [], {}
        for lab in range(n_classes):
            lst = sorted(by_class[lab], key=lambda t: t.ce)
            n = (min(args.per_class_target, len(lst)) if lab in targets
                 else len(lst))
            kept_final.extend(lst[:n])
            per_class[lab] = n

        teachers = sorted({t.teacher_type for t in kept_final})
        per_graph = defaultdict(list)
        for t in kept_final:
            per_graph[t.graph_idx].append(t)
        max_g = max(per_graph.keys(), default=-1)
        result = {tn: [[] for _ in range(max_g + 1)] for tn in teachers}
        for g_idx, lst in per_graph.items():
            for t in lst:
                result[t.teacher_type][g_idx].append(t)

        save = {
            'trajectories': result,
            'config': {
                'dataset': spec.name,
                'budget_ratio': ratio,
                'teachers': teachers,
                'per_class_target': args.per_class_target,
                'target_labels': targets,
                'top_k': args.top_k,
                'sim_threshold': args.sim_threshold,
                'balance_report': {
                    'n_pool': len(pool),
                    'n_after_dedup': n_after_dedup,
                    'dedup_removed': n_dedup,
                    'n_after_top_k': len(kept_by_graph),
                    'n_kept': len(kept_final),
                    'per_class': per_class,
                },
            },
        }
        save_path = output_dir / f'trajectories_r{key}.pt'
        torch.save(save, str(save_path))
        br = save['config']['balance_report']
        balance_report[key] = br
        print(f"\n{key}: pool={br['n_pool']} dedup_removed={br['dedup_removed']} "
              f"top-k={br['n_after_top_k']} kept={br['n_kept']}")
        print("  per-class: " + " ".join(
            f"{lab}:{n}" for lab, n in per_class.items()))
        print(f"Saved: {save_path}")

    report_path = output_dir / 'balance_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({'per_ratio': balance_report,
                   'round_stats': report}, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved: {report_path}")


if __name__ == '__main__':
    main()
