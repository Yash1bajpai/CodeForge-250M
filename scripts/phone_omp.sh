#!/data/data/com.termux/files/usr/bin/bash
# Native Termux entry point for the Debian project's coding agent.
set -euo pipefail

if [[ $# == 0 ]]; then
    set -- --model google-antigravity/gemini-3.8-flash --continue
fi

# Pass argv separately from the fixed shell program, including empty arguments.
exec codeforge /bin/bash -c 'cd -- "$1" && shift && exec /root/.bun/bin/omp "$@"' \
    codeforge-omp /root/projects/own_llm/CodeForge-250M_win "$@"
