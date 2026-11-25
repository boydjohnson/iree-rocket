#!/usr/bin/env bash
# Fails if any ELF file under the given directories requires a GLIBC symbol
# version newer than MAX (first argument), and prints the newest each one needs.
#
#   check-glibc-ceiling.sh 2.39 dist/iree-compile-x86_64 dist/iree-host-tools
#
# readelf rather than objdump: it reads the version-needs section of any
# architecture's ELF, so the aarch64 board binaries are checked from the
# x86_64 runner too. A green build that ships binaries the developer host or
# the board cannot load is the failure this exists to catch (see the
# `runner` input of iree-build.yml).
set -euo pipefail

max="$1"
shift

status=0
checked=0
while IFS= read -r -d '' f; do
  # Not an ELF (scripts, .mlir, .txt): nothing to check.
  readelf -h "$f" >/dev/null 2>&1 || continue
  checked=$((checked + 1))
  need="$(readelf -V --wide "$f" 2>/dev/null | grep -o 'GLIBC_[0-9][0-9.]*' \
            | sed 's/^GLIBC_//' | sort -uV | tail -n1 || true)"
  if [ -z "$need" ]; then
    printf '  %-8s %s\n' "-" "$f"
    continue
  fi
  newest="$(printf '%s\n%s\n' "$need" "$max" | sort -V | tail -n1)"
  if [ "$newest" != "$max" ]; then
    echo "::error file=${f}::requires GLIBC_${need}, newer than the ${max} ceiling"
    status=1
  fi
  printf '  %-8s %s\n' "$need" "$f"
done < <(find "$@" -type f -print0)

if [ "$checked" -eq 0 ]; then
  echo "::error::no ELF files found under: $*"
  exit 1
fi
echo "glibc ceiling ${max}: ${checked} ELF file(s) checked"
exit "$status"
