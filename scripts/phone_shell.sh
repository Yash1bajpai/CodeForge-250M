#!/data/data/com.termux/files/usr/bin/bash
# Native Termux entry point; the coding agent and cloud clients run in Debian.
set -euo pipefail
umask 077
if [[ "${CF_TRAINING_APPROVED:-}" == 1 ]]; then
    printf '%s\n' 'Refusing a controller shell with model-training approval enabled.' >&2
    exit 2
fi
command -v proot-distro >/dev/null
for path in projects codeforge-storage codeforge-migration .kaggle .cache/huggingface .config/codeforge .config/rclone .config/gh .omp; do
    mkdir -p "$HOME/$path"
done
chmod 700 "$HOME/codeforge-migration" "$HOME/.kaggle" "$HOME/.cache/huggingface" "$HOME/.config/codeforge" "$HOME/.config/rclone" "$HOME/.config/gh" "$HOME/.omp"
bindings=()
for path in projects codeforge-storage codeforge-migration .kaggle .cache/huggingface .config/codeforge .config/rclone .config/gh .omp; do
    bindings+=(--bind "$HOME/$path:/root/$path")
done
if [[ $# == 0 ]]; then
    set -- /bin/bash -c 'cd /root/projects/own_llm/CodeForge-250M_win && exec /bin/bash -i'
fi
exec proot-distro login debian "${bindings[@]}" -- /usr/bin/env \
    PATH=/root/.venvs/codeforge/bin:/root/.bun/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_TELEMETRY=1 "$@"
