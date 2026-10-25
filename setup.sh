#!/bin/bash
# Run with: source setup.sh [--migrate-conda]
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Please run: source setup.sh [--migrate-conda]" >&2
    exit 1
fi

_humor_setup() {
    local project_dir env_dir uv_bin backup_path
    project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || return 1
    env_dir="$project_dir/.venv"
    if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != "--migrate-conda" ) ]]; then
        echo "Usage: source setup.sh [--migrate-conda]" >&2
        return 1
    fi
    uv_bin="$(command -v uv || true)"
    if [[ -z "$uv_bin" && -x "$HOME/.local/bin/uv" ]]; then
        uv_bin="$HOME/.local/bin/uv"
    fi
    if [[ -z "$uv_bin" ]]; then
        echo "Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
        return 1
    fi
    if [[ ! -f "$project_dir/uv.lock" ]]; then
        echo "Missing uv.lock. Upload the complete uv update before setup." >&2
        return 1
    fi
    if [[ -f "$env_dir/conda-meta/history" ]]; then
        if [[ "${1:-}" != "--migrate-conda" ]]; then
            echo "Conda .venv detected. Run source setup.sh --migrate-conda to preserve it as a backup." >&2
            return 1
        fi
    elif [[ -e "$env_dir" && ! -f "$env_dir/pyvenv.cfg" ]]; then
        echo "Unrecognized .venv directory; preserved without modification." >&2
        return 1
    fi
    "$uv_bin" python install 3.11 || return 1
    if [[ -f "$env_dir/conda-meta/history" ]]; then
        backup_path="$project_dir/.venv-conda-backup-$(date -u +%Y%m%dT%H%M%SZ)"
        if [[ -e "$backup_path" ]]; then
            echo "Backup path already exists: $backup_path" >&2
            return 1
        fi
        if [[ "${CONDA_PREFIX:-}" == "$env_dir" ]]; then
            conda deactivate || return 1
        fi
        mv -- "$env_dir" "$backup_path" || return 1
        echo "Conda files preserved at: $backup_path"
        echo "A moved conda environment is a backup, not a relocatable runnable environment."
    fi
    UV_PROJECT_ENVIRONMENT="$env_dir" "$uv_bin" sync --project "$project_dir" --locked --python 3.11 --managed-python || return 1
    "$uv_bin" pip check --python "$env_dir/bin/python" || return 1
    "$env_dir/bin/python" "$project_dir/scripts/check_environment.py" \
        --report "$project_dir/.environment.json" || return 1
    source "$env_dir/bin/activate" || return 1
    echo "Setup passed. uv environment: $env_dir"
    echo "Next: cd '$project_dir' && uv run --locked python -m unittest discover -s tests -v"
}

if _humor_setup "$@"; then
    unset -f _humor_setup
else
    unset -f _humor_setup
    echo "Setup did not pass. Keep the error output; do not submit the demo yet." >&2
    return 1
fi
