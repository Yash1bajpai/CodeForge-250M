#!/data/data/com.termux/files/usr/bin/bash
# Install native Termux launchers; Debian's real omp remains untouched.
set -euo pipefail
umask 077

native_prefix=/data/data/com.termux/files/usr
if [[ "${PREFIX:-}" != "$native_prefix" || ! -x "$native_prefix/bin/bash" || -x /usr/bin/apt-get ]]; then
    printf '%s\n' 'Run this installer in native Termux, not inside Debian.' >&2
    exit 2
fi

source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
install -m 755 -- "$source_dir/phone_shell.sh" "$native_prefix/bin/codeforge"
install -m 755 -- "$source_dir/phone_omp.sh" "$native_prefix/bin/omp"
install -m 755 -- "$source_dir/phone_check.sh" "$native_prefix/bin/cf-check"
printf '%s\n' 'Installed native codeforge, omp, and cf-check commands.'
