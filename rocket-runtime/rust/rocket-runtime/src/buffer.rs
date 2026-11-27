use std::ptr::NonNull;

use rocket_runtime_sys as sys;

use crate::{
    Session,
    error::{Error, Result, StatusCode, check},
};

/// The element type of a tensor, IREE's `iree_hal_element_type_t`.
///
/// MLIR integers are signless, so a model's `tensor<...xi8>` is `Int8`, not
/// `Sint8`; that is what `i8` and `u8` slices are created as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ElementType {
    Bool8,
    Int4,
    Int8,
    Int16,
    Int32,
    Int64,
    Sint8,
    Sint16,
    Sint32,
    Sint64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Float16,
    BFloat16,
    Float32,
    Float64,
    /// A type this crate does not name, by its raw IREE value.
    Other(u32),
}

macro_rules! element_types {
    ($($variant:ident => $raw:ident),* $(,)?) => {
        impl ElementType {
            pub fn to_raw(self) -> sys::iree_hal_element_type_t {
                match self {
                    $(ElementType::$variant => sys::$raw as sys::iree_hal_element_type_t,)*
                    ElementType::Other(raw) => raw,
                }
            }

            pub fn from_raw(raw: sys::iree_hal_element_type_t) -> Self {
                $(if raw == sys::$raw as sys::iree_hal_element_type_t {
                    return ElementType::$variant;
                })*
                ElementType::Other(raw)
            }
        }
    };
}

element_types! {
    Bool8 => iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_BOOL_8,
    Int4 => iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_INT_4,
    Int8 => iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_INT_8,
    Int16 => iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_INT_16,
    Int32 => iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_INT_32,
    Int64 => iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_INT_64,
    Sint8 => iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_SINT_8,
    Sint16 => iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_SINT_16,
    Sint32 => iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_SINT_32,
    Sint64 => iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_SINT_64,
    Uint8 => iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_UINT_8,
    Uint16 => iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_UINT_16,
    Uint32 => iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_UINT_32,
    Uint64 => iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_UINT_64,
    Float16 => iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_FLOAT_16,
    BFloat16 => iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_BFLOAT_16,
    Float32 => iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_FLOAT_32,
    Float64 => iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_FLOAT_64,
}

impl ElementType {
    /// Bits per element: the low byte of IREE's encoding
    /// (`iree_hal_element_bit_count`).
    pub fn bit_count(self) -> u32 {
        self.to_raw() & 0xFF
    }

    /// Bytes for `count` densely packed elements, rounding a sub-byte type's
    /// last partial byte up.
    fn dense_byte_count(self, count: usize) -> usize {
        (count * self.bit_count() as usize).div_ceil(8)
    }
}

mod sealed {
    pub trait Sealed {}
}

/// A Rust type whose slices map onto a tensor element type byte for byte.
///
/// # Safety
///
/// Implemented only for plain-old-data types that are valid for every bit
/// pattern of their size, so a tensor's bytes can be read into them.
pub unsafe trait Element: Copy + sealed::Sealed + 'static {
    /// The type a slice of `Self` is created as.
    const ELEMENT_TYPE: ElementType;

    /// Whether a tensor of `element_type` can be read as `Self`: the same
    /// width, and a signless integer reads as either signedness.
    fn reads(element_type: ElementType) -> bool {
        element_type == Self::ELEMENT_TYPE
    }
}

macro_rules! element {
    ($ty:ty => $element:ident $(, also $($other:ident),+)?) => {
        impl sealed::Sealed for $ty {}
        unsafe impl Element for $ty {
            const ELEMENT_TYPE: ElementType = ElementType::$element;
            fn reads(element_type: ElementType) -> bool {
                element_type == ElementType::$element
                    $($(|| element_type == ElementType::$other)+)?
            }
        }
    };
}

element!(f32 => Float32);
element!(f64 => Float64);
element!(i8 => Int8, also Sint8);
element!(u8 => Int8, also Uint8, Bool8);
element!(i16 => Int16, also Sint16);
element!(u16 => Int16, also Uint16);
element!(i32 => Int32, also Sint32);
element!(u32 => Int32, also Uint32);
element!(i64 => Int64, also Sint64);
element!(u64 => Int64, also Uint64);
#[cfg(feature = "half")]
element!(half::f16 => Float16);
#[cfg(feature = "half")]
element!(half::bf16 => BFloat16);

/// A tensor: a shaped, typed view of a HAL buffer (`iree_hal_buffer_view_t`).
///
/// Reference counted; `clone` is cheap and shares the buffer.
pub struct BufferView {
    raw: NonNull<sys::iree_hal_buffer_view_t>,
}

// SAFETY: IREE buffer views are immutable after creation and reference
// counted atomically; reading one from several threads is what the HAL allows.
unsafe impl Send for BufferView {}
unsafe impl Sync for BufferView {}

impl BufferView {
    /// Copies `data` into a new buffer on the session's lead device (the NPU,
    /// or the CPU in a CPU-only session), shaped `shape`.
    pub fn from_slice<T: Element>(session: &Session, shape: &[usize], data: &[T]) -> Result<Self> {
        // SAFETY: Element types are plain old data.
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data))
        };
        Self::from_bytes(session, shape, T::ELEMENT_TYPE, bytes)
    }

    /// As `from_slice`, for element types with no Rust counterpart (f16
    /// without the `half` feature, for one): `bytes` is the dense row-major
    /// tensor data.
    pub fn from_bytes(
        session: &Session,
        shape: &[usize],
        element_type: ElementType,
        bytes: &[u8],
    ) -> Result<Self> {
        let count: usize = shape.iter().product();
        let expected = element_type.dense_byte_count(count);
        if bytes.len() != expected {
            return Err(Error::new(
                StatusCode::InvalidArgument,
                format!(
                    "a {shape:?} {element_type:?} tensor is {expected} bytes, but {} were given",
                    bytes.len()
                ),
            ));
        }
        let dims: Vec<sys::iree_hal_dim_t> =
            shape.iter().map(|&d| d as sys::iree_hal_dim_t).collect();
        let params = sys::iree_hal_buffer_params_t {
            type_: sys::iree_hal_memory_type_bits_t_IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL as _,
            usage: sys::iree_hal_buffer_usage_bits_t_IREE_HAL_BUFFER_USAGE_DEFAULT as _,
            ..Default::default()
        };
        let mut raw = std::ptr::null_mut();
        // SAFETY: the device and allocator belong to the live session; IREE
        // copies `bytes` before returning.
        check(unsafe {
            sys::iree_hal_buffer_view_allocate_buffer_copy(
                session.lead_device(),
                session.lead_allocator(),
                dims.len(),
                dims.as_ptr(),
                element_type.to_raw(),
                sys::iree_hal_encoding_types_t_IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR as _,
                params,
                sys::iree_make_const_byte_span(bytes),
                &mut raw,
            )
        })?;
        // SAFETY: success hands back a retained, non-null view.
        Ok(unsafe { Self::from_raw(raw) })
    }

    /// The dimensions, outermost first.
    pub fn shape(&self) -> Vec<usize> {
        // SAFETY: the view is live, and IREE's dims array has `rank` entries.
        unsafe {
            let rank = sys::iree_hal_buffer_view_shape_rank(self.raw.as_ptr());
            let dims = sys::iree_hal_buffer_view_shape_dims(self.raw.as_ptr());
            std::slice::from_raw_parts(dims, rank)
                .iter()
                .map(|&d| d as usize)
                .collect()
        }
    }

    pub fn element_type(&self) -> ElementType {
        // SAFETY: the view is live.
        ElementType::from_raw(unsafe { sys::iree_hal_buffer_view_element_type(self.raw.as_ptr()) })
    }

    /// The number of elements, the product of the shape.
    pub fn element_count(&self) -> usize {
        self.shape().iter().product()
    }

    /// The size of the tensor's data in bytes.
    pub fn byte_length(&self) -> usize {
        // SAFETY: the view is live.
        unsafe { sys::iree_hal_buffer_view_byte_length(self.raw.as_ptr()) as usize }
    }

    /// Copies the tensor's data out, dense and row-major.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = vec![0u8; self.byte_length()];
        self.read_into(&mut bytes)?;
        Ok(bytes)
    }

    /// Copies the tensor's data out as `T`s. Fails if the element type is
    /// not one `T` reads (see `Element::reads`).
    pub fn to_vec<T: Element>(&self) -> Result<Vec<T>> {
        let element_type = self.element_type();
        if !T::reads(element_type) {
            return Err(Error::new(
                StatusCode::InvalidArgument,
                format!(
                    "the tensor is {element_type:?}; {} reads {:?}",
                    std::any::type_name::<T>(),
                    T::ELEMENT_TYPE
                ),
            ));
        }
        let count = self.byte_length() / std::mem::size_of::<T>();
        let mut values: Vec<T> = Vec::with_capacity(count);
        // SAFETY: the spare capacity is exactly byte_length bytes, written in
        // full by read_into before the length is set; T is valid for any bits.
        unsafe {
            let bytes = std::slice::from_raw_parts_mut(
                values.as_mut_ptr().cast::<u8>(),
                count * std::mem::size_of::<T>(),
            );
            self.read_into(bytes)?;
            values.set_len(count);
        }
        Ok(values)
    }

    fn read_into(&self, bytes: &mut [u8]) -> Result<()> {
        // SAFETY: the view and its buffer are live; IREE writes at most
        // `bytes.len()` bytes, mapping (and invalidating) the buffer for read.
        check(unsafe {
            sys::iree_hal_buffer_map_read(
                sys::iree_hal_buffer_view_buffer(self.raw.as_ptr()),
                0,
                bytes.as_mut_ptr().cast(),
                bytes.len() as _,
            )
        })
    }

    /// The underlying `iree_hal_buffer_view_t`, still owned by `self`.
    pub fn as_raw(&self) -> *mut sys::iree_hal_buffer_view_t {
        self.raw.as_ptr()
    }

    /// Takes ownership of one reference to `raw`.
    ///
    /// # Safety
    ///
    /// `raw` must be a live, non-null buffer view whose reference the caller
    /// hands over.
    pub unsafe fn from_raw(raw: *mut sys::iree_hal_buffer_view_t) -> Self {
        Self {
            raw: NonNull::new(raw).expect("null iree_hal_buffer_view_t"),
        }
    }
}

impl Clone for BufferView {
    fn clone(&self) -> Self {
        // SAFETY: the view is live; the clone owns the new reference.
        unsafe { sys::iree_hal_buffer_view_retain(self.raw.as_ptr()) };
        Self { raw: self.raw }
    }
}

impl Drop for BufferView {
    fn drop(&mut self) {
        // SAFETY: self owns one reference.
        unsafe { sys::iree_hal_buffer_view_release(self.raw.as_ptr()) };
    }
}

impl std::fmt::Debug for BufferView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferView")
            .field("shape", &self.shape())
            .field("element_type", &self.element_type())
            .finish()
    }
}
