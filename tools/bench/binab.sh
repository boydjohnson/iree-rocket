#!/bin/bash
# Runs on the board. A/B of two runtime binaries on the same model files --
# the regression gate for a driver change that is not supposed to move any
# number (MULTICORE.md M0: the worker pool at N=1 must reproduce the
# thread-per-dispatch driver's numbers before N>1 is built on it).
#
# Same protocol as m4sweep.sh: governor and IRQ affinity echoed in the header,
# every NPU core read `suspended` before each run, arms interleaved so drift
# lands evenly, and the binary order alternates each pass so neither is
# always second. Quote medians per (arm, cpuset, bin) across passes.
#
#   A=iree-benchmark-module-pre-m0 B=iree-benchmark-module-m0 \
#     ARMS="mnv2.fp16 mnv2.int8 mnv2.static-int8" ./binab.sh
set -u
cd ~
A=${A:-$HOME/iree-benchmark-module-pre-m0}
B=${B:-$HOME/iree-benchmark-module-m0}
PASSES=${PASSES:-4}
MINTIME=${MINTIME:-3s}
CPUSETS=${CPUSETS:-"4-7 0-7"}
# Model files in ~, without the .vmfb suffix. All take mnv2_in.npy's
# 1x3x224x224 f32 input, ViT included.
ARMS=${ARMS:-"mnv2.fp16 mnv2.int8 mnv2.static-int8 vit.npu.caps3584"}
INPUT=${INPUT:-mnv2_in.npy}

wait_quiet() {
  for _ in $(seq 1 60); do
    busy=0
    for f in /sys/devices/platform/fda*0000.npu/power/runtime_status; do
      [ "$(cat "$f" 2>/dev/null)" = suspended ] || busy=1
    done
    [ "$busy" = 0 ] && return 0
    sleep 0.5
  done
  echo "# WARNING: NPU never settled to suspended"
}

echo "# A:        $A ($(md5sum "$A" | cut -c1-8))"
echo "# B:        $B ($(md5sum "$B" | cut -c1-8))"
echo "# governor: $(for c in 0 4 6; do printf 'cpu%s=%s ' $c \
    "$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor)"; done)"
echo "# npu irqs: $(for i in 82 83 84; do printf '%s->%s ' $i "$(cat /proc/irq/$i/smp_affinity_list)"; done)"
echo "# passes:   $PASSES   min_time: $MINTIME   cpusets: $CPUSETS"
echo "# input:    $INPUT ($(md5sum "$INPUT" | cut -c1-8))"
for a in $ARMS; do echo "# arm $a: $(md5sum $a.vmfb | cut -c1-8)"; done
echo

run_one() {
  local bin=$1 label=$2 arm=$3 cpus=$4 p=$5
  wait_quiet
  sleep 1
  out=$(taskset -c "$cpus" "$bin" \
          --module=$arm.vmfb --device=rocket --device=local-task \
          --function=main_graph --input=@"$INPUT" \
          --benchmark_min_time=$MINTIME 2>&1)
  ips=$(echo "$out" | grep -oE 'items_per_second=[0-9.]+' | cut -d= -f2)
  ms=$(echo "$out" | grep -oE 'real_time[[:space:]]+[0-9.]+ ms' | grep -oE '[0-9.]+')
  hangs=$(echo "$out" | grep -c 'hung-job floor')
  printf '%-18s cpus=%-4s pass%-2s %-3s %8s items/s %9s ms hangs=%s\n' \
      "$arm" "$cpus" "$p" "$label" "${ips:-ABORTED}" "${ms:-?}" "$hangs"
}

for p in $(seq 1 "$PASSES"); do
  for cpus in $CPUSETS; do
    for arm in $ARMS; do
      if [ $((p % 2)) = 1 ]; then
        run_one "$A" A "$arm" "$cpus" "$p"
        run_one "$B" B "$arm" "$cpus" "$p"
      else
        run_one "$B" B "$arm" "$cpus" "$p"
        run_one "$A" A "$arm" "$cpus" "$p"
      fi
    done
  done
done
