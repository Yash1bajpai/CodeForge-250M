#!/usr/bin/env bash
# Run inside the phone's Debian proot, not the native Android shell.
set -euo pipefail
umask 077
if [[ ! -x /usr/bin/apt-get ]]; then
    printf '%s\n' 'Run this script inside the Debian environment, not native Termux.' >&2
    exit 2
fi
if [[ "${CF_TRAINING_APPROVED:-}" == 1 ]]; then
    printf '%s\n' 'Refusing setup with the model-training approval flag enabled.' >&2
    exit 2
fi
if [[ "$(uname -m)" != aarch64 ]]; then
    printf '%s\n' 'This setup targets the ARM64 Debian phone environment.' >&2
    exit 2
fi
command -v apt-get >/dev/null
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl git gh python3 python3-venv python3-pip rclone tmux unzip
mkdir -p "$HOME/.venvs" "$HOME/codeforge-storage"/{datasets,checkpoints,evaluations,logs} "$HOME/.config/codeforge" "$HOME/.config/rclone" "$HOME/.config/gh" "$HOME/.cache/huggingface" "$HOME/.kaggle"
chmod 700 "$HOME/.config/codeforge" "$HOME/.config/rclone" "$HOME/.config/gh" "$HOME/.cache/huggingface" "$HOME/.kaggle"
python3 -m venv "$HOME/.venvs/codeforge"
"$HOME/.venvs/codeforge/bin/python" -m pip install -r "$PROJECT/scripts/requirements-phone.txt"
# Debian's packaged rclone predates the supported Drive authorization flow.
if [[ ! -x /usr/local/bin/rclone ]] || [[ "$(rclone version)" != "rclone v1.75.2"$'\n'* ]]; then
    staging="$(mktemp -d)"
    trap 'rm -rf -- "$staging"' EXIT
    archive="$staging/rclone-v1.75.2-linux-arm64.zip"
    curl -fsSL --connect-timeout 20 --max-time 180 \
        https://downloads.rclone.org/v1.75.2/rclone-v1.75.2-linux-arm64.zip -o "$archive"
    printf '%s  %s\n' 7e1e8d69654941b7b7df84ee74c5f7fb09cce7d5496947fb7bf16b55ff427d10 "$archive" | sha256sum --check -
    unzip -p "$archive" rclone-v1.75.2-linux-arm64/rclone > /usr/local/bin/rclone
    chmod 755 /usr/local/bin/rclone
    rm -rf -- "$staging"
    trap - EXIT
fi
if [[ ! -x "$HOME/.bun/bin/bun" ]] || [[ "$("$HOME/.bun/bin/bun" --version)" != 1.4.3 ]]; then
    installer="$(mktemp)"
    trap 'rm -f -- "$installer"' EXIT
    curl -fsSL --connect-timeout 20 --max-time 180 https://bun.sh/install -o "$installer"
    bash "$installer" bun-v1.4.3
    rm -f -- "$installer"
    trap - EXIT
fi
export PATH="$HOME/.bun/bin:$PATH"
if [[ ! -x "$HOME/.bun/bin/omp" ]] || [[ "$("$HOME/.bun/bin/omp" --version)" != omp/18.12.6 ]]; then
    bun add --global --omit=optional --network-concurrency=8 \
        @oh-my-pi/pi-coding-agent@18.12.6 @oh-my-pi/pi-natives-linux-arm64@18.12.6
fi
printf '%s\n' 'Controller dependencies installed. No job was submitted and no training approval was set.'
printf 'Activate: source %q/bin/activate\n' "$HOME/.venvs/codeforge"
printf 'Project: %s\n' "$PROJECT"
