#!/usr/bin/env bash
# =============================================================================
# One-shot environment setup (conda env "gnn") for Linux servers.
#
# Steps:
#   1. Locate a conda installation
#   2. Create the env from code/environment-linux.yml
#      (torch 2.3.0+cu121, dgl 2.2.1, ogb 1.3.6, pyg 2.7.0)
#   3. Verify the installation (import torch / dgl / ogb / torch_geometric)
#
# Usage:
#   bash setup_env.sh
#
# Prerequisites: Linux with Miniconda/Anaconda installed. First creation
# downloads several GB and can take a while.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_YML="$SCRIPT_DIR/code/environment-linux.yml"

log()  { echo -e "\033[1;32m[OK]\033[0m   $*"; }
info() { echo -e "\033[1;34m[..]\033[0m   $*"; }
die()  { echo -e "\033[1;31m[FAIL]\033[0m $*" >&2; exit 1; }

# ---- 1. locate conda ----
[[ "$(uname -s)" == "Linux" ]] || die "Linux only"
CONDA_ROOT=""
if command -v conda >/dev/null 2>&1 && [[ -x "$(command -v conda)" ]]; then
    CONDA_ROOT="$(cd "$(dirname "$(readlink -f "$(command -v conda)")")/.." && pwd)"
fi
if [[ -z "$CONDA_ROOT" ]]; then
    for cand in "$HOME/miniconda3" "$HOME/anaconda3" /opt/miniconda3 /opt/anaconda3 /opt/conda \
                /home/*/miniconda3 /home/*/anaconda3; do
        if [[ -x "$cand/bin/conda" ]]; then
            CONDA_ROOT="$cand"
            break
        fi
    done
fi
[[ -n "$CONDA_ROOT" ]] || die "conda not found; install Miniconda first (https://docs.conda.io/en/latest/miniconda.html)"
source "$CONDA_ROOT/etc/profile.d/conda.sh" 2>/dev/null || die "cannot source $CONDA_ROOT/etc/profile.d/conda.sh"
command -v conda >/dev/null 2>&1 || die "conda failed to load"
log "conda root: $CONDA_ROOT"

# ---- 2. create the environment ----
[[ -f "$ENV_YML" ]] || die "environment file not found: $ENV_YML"
if conda env list | awk '{print $1}' | grep -qx 'gnn'; then
    log "env 'gnn' already exists, skipping creation (rebuild: conda env remove -n gnn, then rerun)"
else
    info "creating conda env 'gnn' (multi-GB download on first run)..."
    conda env create -f "$ENV_YML"
    log "env 'gnn' created"
fi

# ---- 3. verify ----
info "verifying environment (import check)..."
conda run -n gnn python - <<'EOF'
import torch, dgl, ogb, torch_geometric
print(f"torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
print(f"dgl={dgl.__version__}  ogb={ogb.__version__}  pyg={torch_geometric.__version__}")
EOF
log "environment verified"

echo
log "done. activate with: conda activate gnn"
