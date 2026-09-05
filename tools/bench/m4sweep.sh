#!/bin/bash
# Runs on the board. Measures each offload arm against its like-for-like
# --no-offload baseline at several core allocations, because ISSUES.md M4 and
# P8 together say a deficit quoted without its allocation is arbitrary within
# 2.4x -- so the allocation is a column, not a footnote.
#
# Build the arms on the host first (see README, "The CPU-only baseline"), then
# copy them to ~/bench on the board under the names in ARMS:
#
#   for m in int8 fp16; do
#     cargo run -p rocket-compiler -- compile --input mnv2.$m.mlir #       --llvmcpu-target-triple aarch64-linux-gnu --output $m.landed.vmfb
#     cargo run -p rocket-compiler -- compile --input mnv2.$m.mlir --no-offload #       --llvmcpu-target-triple aarch64-linux-gnu --output $m.cpu.vmfb
#   done
#
# Pin the A76 governors to `performance` and put the NPU IRQs on a big core
# before running; the header echoes both so a result carries its conditions.
set -u
cd ~
BIN=${BIN:-$HOME/iree-benchmark-module}
PASSES=${PASSES:-3}
MINTIME=${MINTIME:-5s}
CPUSETS=${CPUSETS:-"4,5 4-7"}
ARMS=${ARMS:-"int8.cpu int8.landed fp16.cpu fp16.landed"}

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

echo "# bin:      $BIN ($(md5sum "$BIN" | cut -c1-8))"
echo "# governor: $(for c in 0 4 6; do printf 'cpu%s=%s ' $c \
    "$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor)"; done)"
echo "# npu irqs: $(for i in 82 83 84; do printf '%s->%s ' $i "$(cat /proc/irq/$i/smp_affinity_list)"; done)"
echo "# passes:   $PASSES   min_time: $MINTIME   cpusets: $CPUSETS"
for a in $ARMS; do echo "# arm $a: $(md5sum bench/$a.vmfb | cut -c1-8)"; done
echo

for p in $(seq 1 "$PASSES"); do
  for cpus in $CPUSETS; do
    for arm in $ARMS; do
      wait_quiet
      out=$(taskset -c "$cpus" "$BIN" \
              --module=bench/$arm.vmfb --device=rocket --device=local-task \
              --function=main_graph --input=@mnv2_in.npy \
              --benchmark_min_time=$MINTIME 2>&1)
      ips=$(echo "$out" | grep -oE 'items_per_second=[0-9.]+' | cut -d= -f2)
      ms=$(echo "$out" | grep -oE 'real_time[[:space:]]+[0-9.]+ ms' | grep -oE '[0-9.]+')
      hangs=$(echo "$out" | grep -c 'hung-job floor')
      printf '%-12s cpus=%-4s pass%-2s %8s items/s %8s ms hangs=%s\n' \
          "$arm" "$cpus" "$p" "${ips:-ABORTED}" "${ms:-?}" "$hangs"
    done
  done
done
