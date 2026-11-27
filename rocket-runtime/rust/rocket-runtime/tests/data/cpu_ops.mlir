// Fixture for tests/cpu_only.rs. Compiled for VMVX, IREE's portable bytecode
// backend, so one checked-in .vmfb runs CPU-only on any host and on the board.
// Regenerate after an IREE bump (a stale one fails to load, loudly):
//
//   iree-compile cpu_ops.mlir --iree-hal-target-device=local \
//     --iree-hal-local-target-device-backends=vmvx -o cpu_ops.vmvx.vmfb

// a * b + a
func.func @mul_add(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %0 = arith.mulf %a, %b : tensor<4xf32>
  %1 = arith.addf %0, %a : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// Two results: a + a and a * a.
func.func @twice(%a: tensor<2x3xi32>) -> (tensor<2x3xi32>, tensor<2x3xi32>) {
  %0 = arith.addi %a, %a : tensor<2x3xi32>
  %1 = arith.muli %a, %a : tensor<2x3xi32>
  return %0, %1 : tensor<2x3xi32>, tensor<2x3xi32>
}
