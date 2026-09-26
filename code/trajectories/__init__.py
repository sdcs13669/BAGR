"""Trajectory pipeline: teacher generation -> filtering -> balancing.

- core/      library (Trajectory, TrajectoryGenerator, filtering, rollout)
- teachers/  teachers (gradient / info-gain / structural / hybrid)
- top-level CLIs: generate_trajectories.py / filter_trajectories.py /
  balance_trajectories.py"""
