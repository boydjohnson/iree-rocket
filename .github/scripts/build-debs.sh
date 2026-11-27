#!/usr/bin/env bash
# Builds one half of the iree-rocket Debian packages from the CI checkout and
# collects the .debs into OUTDIR:
#
#   build-debs.sh compiler OUTDIR   rocket-compiler, packaging the IREE build
#                                   already in iree-build/build
#                                   (ROCKET_IREE_BUILD_DIR), so no second
#                                   LLVM build
#   build-debs.sh runtime OUTDIR    libiree-rocket-runtime<N>, -dev and
#                                   iree-rocket-tools, built natively
#                                   (arm64 only)
#
# The checked-in changelog names one fixed snapshot. Each nightly .deb gets
# its own version instead, so apt sees every nightly as newer than the last:
# <base>~git<commit time, UTC, to the minute>.<short sha>+iree<IREE>-1. The
# time, not just the date, so two commits on one day still sort by when they
# landed rather than by hash. Only this workspace's copy of debian/changelog
# changes.
#
# Binary-only, and never `debian/rules clean` first (--no-pre-clean): clean
# runs dh_clean over the whole tree, submodules included, and the IREE build
# it would package lives in that tree.
set -euo pipefail

HALF="$1"
OUTDIR="$(realpath -m "$2")"
ROOT="$(git rev-parse --show-toplevel)"
cd "${ROOT}"

case "${HALF}" in
  compiler)
    PROFILE=pkg.iree-rocket.noruntime
    export ROCKET_IREE_BUILD_DIR="${ROOT}/iree-build/build"
    ;;
  runtime)
    PROFILE=pkg.iree-rocket.nocompiler
    ;;
  *)
    echo "usage: $0 compiler|runtime OUTDIR" >&2
    exit 2
    ;;
esac

# IREE's version from the same file the runtime's SONAME is derived from
# (rocket-runtime/CMakeLists.txt), so the two cannot disagree. "3.14.0.dev"
# becomes "3.14.0".
IREE_VERSION="$(python3 -c '
import json, re, sys
v = json.load(open(sys.argv[1]))["package-version"]
print(re.match(r"\d+\.\d+\.\d+", v).group(0))
' iree-build/iree-src/runtime/version.json)"
COMMIT_TIME="$(TZ=UTC git log -1 --format=%cd --date=format-local:%Y%m%d%H%M)"
SHORT_SHA="$(git rev-parse --short=7 HEAD)"
OLD_VERSION="$(dpkg-parsechangelog -S Version)"
BASE="${OLD_VERSION%%[~+-]*}"
NEW_VERSION="${BASE}~git${COMMIT_TIME}.${SHORT_SHA}+iree${IREE_VERSION}-1"

python3 - "${OLD_VERSION}" "${NEW_VERSION}" <<'EOF'
import sys
old, new = sys.argv[1], sys.argv[2]
path = "debian/changelog"
lines = open(path).read().split("\n")
head = f"({old})"
if head not in lines[0]:
    sys.exit(f"{path}: first line does not carry {head}: {lines[0]!r}")
lines[0] = lines[0].replace(head, f"({new})", 1)
open(path, "w").write("\n".join(lines))
EOF
echo "version: ${OLD_VERSION} -> ${NEW_VERSION}"

# The packages land in the checkout's parent directory, which holds nothing
# else of ours: clear any from an earlier run so the collection below takes
# only this build's.
rm -f ../*.deb ../*.ddeb ../*.changes ../*.buildinfo

# -d: Build-Depends name the distribution's cargo and rustc, and CI uses the
# pinned rustup toolchain instead.
DEB_BUILD_OPTIONS="parallel=$(nproc) ${DEB_BUILD_OPTIONS:-}" \
  dpkg-buildpackage -b -us -uc -d --no-pre-clean -P"${PROFILE}"

mkdir -p "${OUTDIR}"
mv ../*.deb "${OUTDIR}/"
for deb in "${OUTDIR}"/*.deb; do
  echo "== $(basename "${deb}") ($(du -h "${deb}" | cut -f1))"
  dpkg-deb -f "${deb}" Package Version Architecture Depends
done
