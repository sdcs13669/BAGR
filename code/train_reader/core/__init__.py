"""train_reader core: unified BC/RL training loops, evaluation loops, model building."""

from .bc import train_one_epoch as bc_train_epoch  # noqa: F401
from .rl import train_one_epoch as rl_train_epoch  # noqa: F401
from .eval_loop import (evaluate_reader, evaluate_fixed,  # noqa: F401
                        build_eval_json, budget_stats, rollout_strategy,
                        evaluate_classifier_full, merge_eval_json)
from .eval_common import (add_common_args, run_split_eval,  # noqa: F401
                          resolve as eval_resolve)
from .build import (load_classifier_frozen, load_reader_from_checkpoint,  # noqa: F401
                    bootstrap_reader)
from .common import (compute_reward, compute_soft_both_reward,  # noqa: F401
                     normalize_advantages, freeze_shared_layers)
