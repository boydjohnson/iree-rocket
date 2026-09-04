# Compatibility policy

## Ownership

`schema/rocket_executable_def.fbs` owns the binary structure and enum values.
The C++ compiler and Rust runtime must use generated bindings rather than
duplicating field offsets, tag values, or byte order.

The schema does not own Rocket hardware limits. Runtime semantic validation
remains authoritative because malformed or unsupported executables must never
reach register-command construction.

## Versioning

`RKT1` is the major wire-format identifier.

Changes that FlatBuffers readers can safely ignore are compatible within
`RKT1`. Examples include adding a field to the end of a table with a safe
default or adding a new member to `KernelDef`.

Changes that alter existing field meaning, scalar type, enum value, or default
require a new file identifier such as `RKT2` and a new HAL executable format
string. Fields and enum values must not be removed or reused.

`Conv2DDef.pad_top` and `pad_left` were added compatibly with a zero default.
Readers interpret an older `RKT1` executable that omits them as an unpadded
convolution. Each leading value is applied symmetrically to the trailing side,
matching the current two-value Rocket `ConvShape` model.

`Precision.INT8_ACCUMULATOR` was appended as enum value 2. Existing INT8 and
FP16 values retain their wire encodings; older runtimes reject the unknown
value rather than interpreting it as a different precision.

`PoolingDef` and `MatmulDef` were appended to `KernelDef` as union tags 3 and
4. `Conv2DDef` keeps tag 1 and `FullyConnectedDef` keeps tag 2. A union tag
is wire-format ABI -- an executable built by an older compiler names its
kernel by that number -- so the members must never be reordered, and
`kernel_union_tags_are_stable` in `tests/compiler_fixture.rs` is what says so.

`FullyConnectedDef` is **deprecated but retained**. No compiler ever emitted
it: "fully connected" is not an operation in MLIR's `linalg` dialect, which
is what IREE hands this backend, and what arrives is `linalg.matmul`.
`MatmulDef` is that operation. Producers must stop emitting
`FullyConnectedDef`; consumers should keep decoding it until no old vmfb
artifacts remain, and must not reuse its tag when they stop.

`PoolingMethod` has no `SUM`. The PPU has no sum mode -- its average is a
multiply by a per-axis reciprocal held as `fp16(65536/k)`, and a divisor of
one would need 65536, past fp16's 65504 ceiling. Since `linalg` has no
average pool either (an ONNX `AveragePool` arrives as `linalg.pooling_*_sum`
plus a separate divide), recognizing that pair as an average is the
compiler's job. Adding `SUM` here would create a wire state no runtime can
execute.

`PoolingDef` deliberately omits the pad fill value that `PoolingShape`
carries. It is the reduction's identity element and follows from `method`
and `precision`, so the runtime derives it; a wire field would let a
producer send a max pool a fill of zero and silently change its answer at
the border.

The IREE `iree_flatbuffer_file_header_t` version remains `0`. The FlatBuffer
file identifier carries the Rocket format version.

## Runtime dimensions

`Conv2DDef.runtime_dimensions` is an optional ordered mapping from dispatch
push-constant ordinals to logical shape fields. Existing executables omit the
vector and remain fully static.

Each vector entry consumes one 32-bit push constant. The corresponding scalar
field in `Conv2DDef` must be zero and is replaced with the nonzero runtime
value before semantic validation. Entries must be known and unique, and the
runtime must reject missing, extra, zero, or hardware-invalid values.

The vector supports input/output width, height, channels, and filter width and
height. Batch remains fixed at one; stride, dilation, depthwise mode, numeric
parameters, and precision remain executable properties.

`PoolingDef.runtime_dimensions` and `MatmulDef.runtime_dimensions` follow the
same contract with their own enums. `PoolingDimension` covers input width and
height, channels, kernel width and height, and both strides -- but **not
padding**, which is 0..=7 on this hardware, means different things per pooling
method, and is not varied per dispatch by any measured model.
`MatmulDimension` covers `M`, `K` and `N`.

The three enums are separate types on purpose. They index different tables and
their numeric values are independent; nothing may assume, for example, that
`INPUT_WIDTH` is 0 in more than one of them because it happens to be so today.

## Export ordinals

`ExecutableDef.exports` is in canonical IREE export ordinal order. The
compiler must reject missing or duplicate ordinals while serializing. The
runtime must require the FlatBuffer export count to equal the executable
layout export count.

`ExportDef.name` is diagnostic metadata. Dispatch selects an export by ordinal,
not by name.

## Migration from the manual format

The existing `rocket-conv2d-v1` payload is:

```text
[tag 3][84-byte little-endian ConvShape record]
```

It describes only one convolution and relies on separately maintained C++ and
Rust layouts. `RKT1` replaces that payload; it is not byte-compatible with it.

During migration, use a distinct HAL executable format string so the driver
can route old and new payloads explicitly. Remove the legacy decoder only
after old VMFB artifacts are no longer supported.

## Required tests

Consumers should share binary golden fixtures produced from this schema:

- C++ producer to Rust verifier/reader.
- Rust producer to FlatCC verifier/reader, if Rust serialization is needed.
- Multiple exports preserve ordinal order.
- Every enum value maps to the expected runtime value.
- Missing required fields, invalid unions, and truncated buffers are rejected.
- Structurally valid but hardware-invalid convolution shapes are rejected by
  runtime semantic validation.
- Runtime dimension mappings reject unknown or duplicate fields and dispatches
  reject missing, extra, zero, or hardware-invalid push constants.
