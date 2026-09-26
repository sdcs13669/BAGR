"""Trajectory filtering: similarity dedup -> per-graph top-k + teacher diversity.

Thin CLI over trajectories/core/filtering.py; output feeds train_bc.py
--data-dir."""

import argparse
import json
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_DIR))
sys.path = [p for p in sys.path
            if Path(p).resolve() != Path(__file__).resolve().parent]

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from tools.paths import CODE_DIR  # noqa: E402
from trajectories.core.filtering import filter_file  # noqa: E402


def ratio_key_from_name(fname: str) -> str:
    """'trajectories_r0_050.pt' / 'trajectories_r0_050_shard1of4.pt' -> '0_050'。"""
    stem = Path(fname).stem
    tail = stem.rsplit('_r', 1)[1]
    return tail.split('_shard')[0]


def resolve(p: str) -> Path:
    from tools.paths import resolve as _resolve
    return _resolve(p)


def main():
    parser = argparse.ArgumentParser(
        description='trajectory filtering (dedup -> per-graph top-k + teacher diversity)')
    parser.add_argument('--input-dir', required=True,
                        help='input directory (generate_trajectories.py output)')
    parser.add_argument('--output-dir', required=True,
                        help='output directory (pass as train_bc.py --data-dir)')
    parser.add_argument('--sim-threshold', type=float, default=0.85,
                        help='cosine similarity threshold for within-graph dedup')
    parser.add_argument('--top-k', type=int, default=3,
                        help='max trajectories kept per graph after dedup')
    parser.add_argument('--no-diversity', action='store_true',
                        help='disable teacher-type diversity (enabled by default)')
    parser.add_argument('--teachers', nargs='+', default=None,
                        help='teacher names (default: each file config.teachers)')
    args = parser.parse_args()

    input_dir = resolve(args.input_dir)
    output_dir = resolve(args.output_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"input directory not found: {input_dir}; run generate_trajectories.py first")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob('trajectories_r*.pt'))
    if not files:
        raise SystemExit(f"no trajectories_r*.pt in {input_dir}")
    require_diversity = not args.no_diversity

    report = {}
    for fpath in tqdm(files, desc="Filtering"):
        key = ratio_key_from_name(fpath.name)
        out = filter_file(fpath, args.teachers, args.top_k, require_diversity,
                          args.sim_threshold)
        report.setdefault(key, []).append(out['report'])
        save_path = output_dir / fpath.name
        torch.save(out['data'], str(save_path))
        r = out['report']
        print(f"  {fpath.name}: {r['n_before']} -> dedup {r['n_after_dedup']} "
              f"(removed {r['dedup_removed']}) -> top-k {r['n_after']} "
              f"(graphs {r['n_graphs_after']}, diverse {r['diverse_graphs']}, "
              f"single {r['single_teacher_graphs']})")

    report_path = output_dir / 'filter_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved: {report_path}")
    print(f"next: python code/train_reader/train_bc.py --data-dir {args.output_dir}")


if __name__ == '__main__':
    main()
