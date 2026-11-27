#!/usr/bin/env bash
# Builds <source>_<upstream>.orig.tar.xz (iree-rocket_...) from this checkout,
# for a 3.0 (quilt) source package that builds with no network:
#
#   - this repository at HEAD, minus debian/ (which ships as debian.tar);
#   - IREE (iree-build/iree-src) at the commit this repository records, with
#     only the IREE submodules the compiler build reads, each at the commit
#     IREE records: llvm-project, torch-mlir, flatcc, benchmark, googletest.
#     The compiler build needs all five and the runtime build (tests off)
#     the last three. The rest (Vulkan, HIP, tracy, stablehlo, ...) are for
#     backends and tools debian/rules turns off; torch-mlir's own nested externals/ are
#     IREE's to replace and are left out too;
#   - every crate in Cargo.lock, vendored under vendor/ (debian/rules finds
#     it and builds --offline).
#
# Everything comes out of git objects (`git archive` of recorded commits),
# so uncommitted edits and whatever happens to be checked out in a submodule
# never leak in. The submodules only have to be initialized.
#
# Usage: debian/make-orig.sh [OUTDIR]          (default: the parent directory)
#        debian/make-orig.sh --stage-only DIR  (unpacked tree, no tarball)
# Env:   IREE_REPO  an initialized iree-src checkout to read IREE's objects
#                   from, when this checkout's submodule is not initialized
#        REV        the commit to package (default HEAD); `git stash create`
#                   gives one holding uncommitted edits, for a trial build
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "${ROOT}"

STAGE_ONLY=
if [ "${1:-}" = "--stage-only" ]; then
  STAGE_ONLY=yes
  shift
fi
OUTDIR="$(realpath -m "${1:-${ROOT}/..}")"

SOURCE="$(dpkg-parsechangelog -S Source)"
VERSION="$(dpkg-parsechangelog -S Version)"
UPSTREAM="${VERSION%-*}"
UPSTREAM="${UPSTREAM#*:}"
NAME="${SOURCE}-${UPSTREAM}"

IREE_SUBMODULES=(llvm-project torch-mlir flatcc benchmark googletest)

# The commit `repo` records for a gitlink: `path` (at HEAD) or `rev:path`.
recorded() {
  case "$2" in
    *:*) git -C "$1" rev-parse "$2" ;;
    *)   git -C "$1" rev-parse "HEAD:$2" ;;
  esac
}

# Unpack `commit` of the git repository at `repo` into `dest`.
export_tree() {
  local repo="$1" commit="$2" dest="$3"
  if ! git -C "${repo}" cat-file -e "${commit}^{commit}" 2>/dev/null; then
    echo "error: ${repo} lacks ${commit}; run" >&2
    echo "  git submodule update --init iree-build/iree-src" >&2
    echo "  git -C iree-build/iree-src submodule update --init ${IREE_SUBMODULES[*]/#/third_party/}" >&2
    exit 1
  fi
  mkdir -p "${dest}"
  git -C "${repo}" archive --format=tar "${commit}" | tar -x -C "${dest}"
}

if [ -n "${STAGE_ONLY}" ]; then
  STAGE="${OUTDIR}"
  [ -e "${STAGE}" ] && { echo "error: ${STAGE} already exists" >&2; exit 1; }
else
  WORK="$(mktemp -d)"
  trap 'rm -rf "${WORK}"' EXIT
  STAGE="${WORK}/${NAME}"
fi
mkdir -p "${STAGE}"

REV="${REV:-HEAD}"
echo "${SOURCE} $(git rev-parse --short "${REV}") -> ${STAGE}"
git archive --format=tar "${REV}" -- . ':(exclude)debian' | tar -x -C "${STAGE}"

IREE="iree-build/iree-src"
# Where IREE's git objects are. A git worktree of this repository usually has
# no submodules of its own; point this at the main checkout's iree-src then.
IREE_REPO="$(realpath "${IREE_REPO:-${IREE}}")"
# An uninitialized submodule is an empty directory, and git -C on it would
# quietly answer for this repository instead.
if [ "$(git -C "${IREE_REPO}" rev-parse --show-toplevel 2>/dev/null)" != "${IREE_REPO}" ]; then
  echo "error: ${IREE_REPO} is not an initialized IREE checkout" >&2
  echo "  git submodule update --init ${IREE}   (or set IREE_REPO)" >&2
  exit 1
fi
IREE_SHA="$(recorded . "${REV}:${IREE}")"
echo "iree ${IREE_SHA}"
export_tree "${IREE_REPO}" "${IREE_SHA}" "${STAGE}/${IREE}"

for sub in "${IREE_SUBMODULES[@]}"; do
  path="third_party/${sub}"
  sha="$(recorded "${IREE_REPO}" "${IREE_SHA}:${path}")"
  echo "iree/${path} ${sha}"
  export_tree "${IREE_REPO}/${path}" "${sha}" "${STAGE}/${IREE}/${path}"
done

echo "vendoring crates"
cargo vendor --quiet --locked --manifest-path "${STAGE}/Cargo.toml" "${STAGE}/vendor" >/dev/null

if [ -n "${STAGE_ONLY}" ]; then
  echo "staged ${STAGE}"
  exit 0
fi

TARBALL="${OUTDIR}/${SOURCE}_${UPSTREAM}.orig.tar.xz"
echo "compressing -> ${TARBALL}"
tar -C "${WORK}" --sort=name --owner=0 --group=0 --numeric-owner \
  --mtime="@$(git log -1 --format=%ct "${REV}")" -cf - "${NAME}" | xz -T0 -6 > "${TARBALL}"
ls -lh "${TARBALL}"
