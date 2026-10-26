#!/data/data/com.termux/files/usr/bin/bash
# Run the consolidated controller check without entering Debian manually.
set -euo pipefail

exec codeforge /root/.venvs/codeforge/bin/python \
    /root/projects/own_llm/CodeForge-250M_win/scripts/phone_control.py check "$@"
