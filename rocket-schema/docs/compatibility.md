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

The IREE `iree_flatbuffer_file_header_t` version remains `0`. The FlatBuffer
file identifier carries the Rocket format version.

## Runtime Conv2D dimensions

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
