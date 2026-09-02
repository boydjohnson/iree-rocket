#!/usr/bin/env bash
# Surveys RK3588 NPU job hangs in an oracle sweep.
#
# Two things this exists to stop us getting wrong.
#
# 1. A hang is not a property of the shape. Starting the same sweep one case
#    later moves the failure to the next case and leaves its *position in the
#    run* unchanged, so failures are scored by position (`--resume-at`), and
#    every hang measured so far has landed on an even position, never an odd
#    one. `ROCKET_PROBE_ONLY` clears a shape partly by making it position 1.
#
# 2. A hang is not a property of the run either -- it depends on what the
#    machine was doing in the second before it. Measured on planck 2026-09-02,
#    same binary, same test, 8-10 runs per row:
#
#      preamble                        failures (cases 6 and 8 of 27)
#      sleep 1 / sleep 2               0/10   -- never
#      sleep 0.1 / 0.25 / 0.5          6-8 of 8
#      back-to-back                    ~4/10
#      2 s CPU burn, no I/O            10/10
#      dd 200 MB + sync                10/10
#      cargo test --no-run             10/10
#      cargo nextest (runs cargo)      10/10
#
#    So `cargo nextest` reproduces near-deterministically and a bare binary on
#    an idle board barely reproduces at all -- same code, same device. It is
#    not --test-threads (1 either way) and only marginally output capture
#    (3/10 nocapture vs 5/10 captured). The mechanism inside the SoC is not
#    established: the NPU cores autosuspend at 50 ms, the CPU governor is
#    ondemand with a 200 ms sample, and CPUs are still at max clock 1 s after a
#    burn -- which does not by itself explain the cliff between 0.5 s and 1 s.
#
# Practical consequence: any gate result is only meaningful next to its
# preamble. Use --matrix to get all four at once.
#
#   tools/npu_hang_survey.sh --matrix --runs 8
#   tools/npu_hang_survey.sh --mode raw --resume-at 0,1,2,3,4,5,6,7
#   tools/npu_hang_survey.sh --mode raw --preamble 'sleep 1' --runs 20
set -euo pipefail

TEST_FN=fp16_mobilenetv2_remaining_pointwise_matches_oracle
TEST_TARGET=conv2d_oracle_hw
PACKAGE=iree-rocket-hal
RUNS=10
MODES="raw"
RESUMES="0"
BINARY=""
THREADS=1
PREAMBLE="true"
MATRIX=0

usage() { sed -n '2,40p' "$0"; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --test) TEST_FN="$2"; shift 2 ;;
        --runs) RUNS="$2"; shift 2 ;;
        --mode) case "$2" in both) MODES="raw nextest" ;; *) MODES="$2" ;; esac; shift 2 ;;
        --resume-at) RESUMES="$(echo "$2" | tr ',' ' ')"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --binary) BINARY="$2"; shift 2 ;;
        --preamble) PREAMBLE="$2"; shift 2 ;;
        --matrix) MATRIX=1; shift ;;
        -h|--help) usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

# `cargo test --no-run` prints one JSON line per artifact; the test binary is
# the last one carrying a non-null executable for this target.
if [ -z "${BINARY}" ]; then
    BINARY=$(cargo test -p "${PACKAGE}" --release --test "${TEST_TARGET}" \
        --no-run --message-format=json 2>/dev/null |
        python3 -c '
import json, sys
path = ""
for line in sys.stdin:
    try:
        message = json.loads(line)
    except ValueError:
        continue
    if message.get("executable"):
        path = message["executable"]
print(path)
')
fi
[ -x "${BINARY}" ] || { echo "no test binary: ${BINARY:-<none>}" >&2; exit 1; }

cpu_burn() {
    local core end
    for core in $(seq "$(nproc)"); do
        ( end=$((SECONDS + 2)); while [ "${SECONDS}" -lt "${end}" ]; do :; done ) &
    done
    wait
}

# One configuration: RUNS invocations, each preceded by `preamble`.
survey() {
    local label="$1" mode="$2" resume="$3" preamble="$4"
    local cases="" timeouts=0 failures=0 run output

    for run in $(seq 1 "${RUNS}"); do
        eval "${preamble}" >/dev/null 2>&1 || true
        if [ "${mode}" = "raw" ]; then
            output=$(ROCKET_PROBE_RESUME_AT="${resume}" "${BINARY}" "${TEST_FN}" \
                --ignored --test-threads="${THREADS}" 2>&1 || true)
        else
            output=$(ROCKET_PROBE_RESUME_AT="${resume}" cargo nextest run --no-fail-fast \
                --release -j1 --test "${TEST_TARGET}" -- \
                --include-ignored "${TEST_FN}" 2>&1 || true)
        fi
        # nextest indents the captured child output; tolerate leading space.
        cases="${cases} $(echo "${output}" |
            sed -n 's|^[[:space:]]*\[\([0-9]*\)/[0-9]*\] FAIL.*|\1|p' | tr '\n' ' ')"
        timeouts=$((timeouts + $(echo "${output}" |
            grep -c 'DEVICE TIMEOUT: that dispatch' || true)))
    done

    failures=$(echo ${cases} | wc -w)
    printf '  %-22s resume_at=%-2s failures=%-3s timeouts=%-3s' \
        "${label}" "${resume}" "${failures}" "${timeouts}"
    if [ "${failures}" -eq 0 ]; then
        echo ' (none)'
        return
    fi
    printf ' case:count(pos) '
    echo ${cases} | tr ' ' '\n' | grep -v '^$' | sort -n | uniq -c |
        awk -v r="${resume}" '{printf "%s:%s(%s) ", $2, $1, $2 - r}'
    echo ${cases} | tr ' ' '\n' | grep -v '^$' |
        awk -v r="${resume}" '{ if (($1 - r) % 2 == 0) even++; else odd++ }
             END { printf " [even=%d odd=%d]\n", even + 0, odd + 0 }'
}

echo "binary : ${BINARY}"
echo "test   : ${TEST_FN}"
echo "runs   : ${RUNS} per configuration"
echo

if [ "${MATRIX}" -eq 1 ]; then
    survey 'idle 1s'      raw     0 'sleep 1'
    survey 'back-to-back' raw     0 'true'
    survey 'cpu burn 2s'  raw     0 'cpu_burn'
    survey 'nextest'      nextest 0 'true'
    exit 0
fi

for mode in ${MODES}; do
    for resume in ${RESUMES}; do
        survey "${mode}" "${mode}" "${resume}" "${PREAMBLE}"
    done
done
