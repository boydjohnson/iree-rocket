use std::{
    alloc::Layout,
    ffi::{CString, c_void},
    marker::PhantomData,
    path::Path,
    ptr::NonNull,
};

use rocket_runtime_sys as sys;

use crate::{
    BufferView,
    error::{Error, Result, StatusCode, check},
};

/// The IREE runtime instance: the VM and the driver registry, with every
/// driver the library carries (rocket, local-task, local-sync) registered.
///
/// Create one per process and share it; sessions keep it alive themselves.
pub struct Instance {
    raw: NonNull<sys::iree_runtime_instance_t>,
}

// SAFETY: an IREE runtime instance is thread-safe; sessions on different
// threads may share it.
unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}

impl Instance {
    pub fn new() -> Result<Self> {
        let mut options = sys::iree_runtime_instance_options_t::default();
        let mut raw = std::ptr::null_mut();
        // SAFETY: options is initialized before use; IREE retains nothing of it.
        check(unsafe {
            sys::iree_runtime_instance_options_initialize(&mut options);
            sys::iree_runtime_instance_options_use_all_available_drivers(&mut options);
            sys::iree_runtime_instance_create(&options, sys::iree_allocator_system(), &mut raw)
        })?;
        Ok(Self {
            raw: NonNull::new(raw).expect("iree_runtime_instance_create returned null"),
        })
    }

    pub fn as_raw(&self) -> *mut sys::iree_runtime_instance_t {
        self.raw.as_ptr()
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        // SAFETY: self owns one reference.
        unsafe { sys::iree_runtime_instance_release(self.raw.as_ptr()) };
    }
}

/// Which devices a session creates.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Devices {
    /// The NPU (device 0, the lead) and the CPU: what every rocket-compiler
    /// module that offloads anything needs. Session creation fails if the NPU
    /// cannot be opened.
    #[default]
    NpuAndCpu,
    /// The CPU alone: enough for a module compiled with `--no-offload`, and
    /// the only choice on a machine without the NPU.
    CpuOnly,
}

/// Which CPUs the CPU device's worker threads run on.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum CpuPlacement {
    /// One worker per CPU sharing the machine's highest `cpu_capacity`
    /// (cpu4-7, the Cortex-A76s, on the RK3588); IREE's default where
    /// capacities are uniform or unpublished. IREE's own placement put
    /// MobileNetV2 fp16 at 105 ms against 60-63 ms pinned to the big cores.
    #[default]
    Fastest,
    /// One worker per listed logical CPU.
    Cpus(Vec<u32>),
    /// IREE's own selection, which honours `--task_topology_*` flags if the
    /// process parsed IREE flags.
    IreeDefault,
}

/// How to build a [`Session`]; the default is the NPU plus the CPU, workers
/// on the fastest cores.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionOptions {
    pub devices: Devices,
    pub cpu_placement: CpuPlacement,
}

impl SessionOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn devices(mut self, devices: Devices) -> Self {
        self.devices = devices;
        self
    }

    pub fn cpu_placement(mut self, placement: CpuPlacement) -> Self {
        self.cpu_placement = placement;
        self
    }
}

/// A VM context with the HAL module bound to the session's devices, plus the
/// modules loaded into it (`rocket_runtime_session_t`).
///
/// `Send` but not `Sync`: IREE contexts are thread-compatible, so one session
/// serves one thread at a time. Use a session per thread to run concurrently.
pub struct Session {
    raw: NonNull<sys::rocket_runtime_session_t>,
}

// SAFETY: a session may move between threads; it is only !Sync.
unsafe impl Send for Session {}

impl Session {
    pub fn new(instance: &Instance, options: &SessionOptions) -> Result<Self> {
        // SAFETY: plain struct initialization.
        let mut raw_options = sys::rocket_runtime_session_options_t::default();
        unsafe { sys::rocket_runtime_session_options_initialize(&mut raw_options) };
        raw_options.devices = match options.devices {
            Devices::NpuAndCpu => sys::rocket_runtime_devices_e_ROCKET_RUNTIME_DEVICES_NPU_AND_CPU,
            Devices::CpuOnly => sys::rocket_runtime_devices_e_ROCKET_RUNTIME_DEVICES_CPU_ONLY,
        } as _;
        // Kept alive across the call: the C side reads it during creation only.
        let cpu_ids;
        raw_options.cpu_policy = match &options.cpu_placement {
            CpuPlacement::Fastest => {
                sys::rocket_runtime_cpu_policy_e_ROCKET_RUNTIME_CPU_POLICY_FASTEST
            }
            CpuPlacement::IreeDefault => {
                sys::rocket_runtime_cpu_policy_e_ROCKET_RUNTIME_CPU_POLICY_IREE_DEFAULT
            }
            CpuPlacement::Cpus(cpus) => {
                if cpus.is_empty() {
                    return Err(Error::new(
                        StatusCode::InvalidArgument,
                        "CpuPlacement::Cpus needs at least one CPU",
                    ));
                }
                let list = cpus
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(",");
                cpu_ids = CString::new(list).expect("digits and commas");
                raw_options.cpu_ids = cpu_ids.as_ptr();
                sys::rocket_runtime_cpu_policy_e_ROCKET_RUNTIME_CPU_POLICY_LIST
            }
        } as _;

        let mut raw = std::ptr::null_mut();
        // SAFETY: the instance is live and the session retains it; options
        // and cpu_ids outlive the call.
        check(unsafe {
            sys::rocket_runtime_session_create(
                instance.as_raw(),
                &raw_options,
                sys::iree_allocator_system(),
                &mut raw,
            )
        })?;
        Ok(Self {
            raw: NonNull::new(raw).expect("rocket_runtime_session_create returned null"),
        })
    }

    /// Loads a `.vmfb` from disk. A module must come from the
    /// rocket-compiler (and so the IREE version) this library was built with.
    pub fn load_module_file(&mut self, path: impl AsRef<Path>) -> Result<()> {
        use std::os::unix::ffi::OsStrExt;
        let path = CString::new(path.as_ref().as_os_str().as_bytes()).map_err(|_| {
            Error::new(
                StatusCode::InvalidArgument,
                "module path contains a NUL byte",
            )
        })?;
        // SAFETY: the session is live; the path outlives the call.
        check(unsafe {
            sys::rocket_runtime_session_append_bytecode_module_from_file(
                self.raw.as_ptr(),
                path.as_ptr(),
            )
        })
    }

    /// Loads a `.vmfb` from memory. The bytes are copied into page-aligned
    /// storage, as IREE's own file loader uses, which the module frees when it
    /// is destroyed.
    pub fn load_module_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        let owned = AlignedBytes::copy_of(bytes);
        let span = sys::iree_const_byte_span_t {
            data: owned.ptr.as_ptr(),
            data_length: owned.layout.size(),
        };
        let deallocator = sys::iree_allocator_t {
            self_: Box::into_raw(Box::new(owned)).cast(),
            ctl: Some(free_aligned_bytes),
        };
        // SAFETY: the session is live. IREE consumes the data even on
        // failure, freeing it through `deallocator`, which owns the Box.
        check(unsafe {
            sys::rocket_runtime_session_append_bytecode_module_from_memory(
                self.raw.as_ptr(),
                span,
                deallocator,
            )
        })
    }

    /// Number of devices: 2 for [`Devices::NpuAndCpu`], 1 for [`Devices::CpuOnly`].
    pub fn device_count(&self) -> usize {
        // SAFETY: the session is live.
        unsafe { sys::rocket_runtime_session_device_count(self.raw.as_ptr()) }
    }

    /// Resolves an exported function, such as `"module.main_graph"`, into a
    /// reusable [`Function`].
    pub fn function(&self, name: &str) -> Result<Function<'_>> {
        let mut call = sys::rocket_runtime_call_t::default();
        // SAFETY: the session is live; the name is only read during the call.
        check(unsafe {
            sys::rocket_runtime_call_initialize_by_name(
                self.raw.as_ptr(),
                sys::iree_make_string_view(name),
                &mut call,
            )
        })
        .map_err(|e| {
            // IREE says only that the name is malformed or missing; the usual
            // cause is leaving off the module part, which ONNX imports name
            // `module`.
            if name.contains('.') {
                e
            } else {
                Error::new(
                    e.code(),
                    format!("{e}\nfunction names are module-qualified: try \"module.{name}\""),
                )
            }
        })?;
        Ok(Function {
            call,
            _session: PhantomData,
        })
    }

    /// Calls `name` once: resolves it, invokes it with `inputs` and returns
    /// its results. Use [`Session::function`] to call one function repeatedly.
    pub fn call(&self, name: &str, inputs: &[&BufferView]) -> Result<Vec<BufferView>> {
        self.function(name)?.invoke(inputs)
    }

    /// Asks every module to drop caches it can rebuild.
    pub fn trim(&mut self) -> Result<()> {
        // SAFETY: the session is live.
        check(unsafe { sys::rocket_runtime_session_trim(self.raw.as_ptr()) })
    }

    pub fn as_raw(&self) -> *mut sys::rocket_runtime_session_t {
        self.raw.as_ptr()
    }

    pub(crate) fn lead_device(&self) -> *mut sys::iree_hal_device_t {
        // SAFETY: the session is live and always has device 0.
        unsafe { sys::rocket_runtime_session_device(self.raw.as_ptr(), 0) }
    }

    pub(crate) fn lead_allocator(&self) -> *mut sys::iree_hal_allocator_t {
        // SAFETY: the session is live.
        unsafe { sys::rocket_runtime_session_device_allocator(self.raw.as_ptr()) }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // SAFETY: self owns one reference.
        unsafe { sys::rocket_runtime_session_release(self.raw.as_ptr()) };
    }
}

/// One resolved function with reusable argument and result lists
/// (`rocket_runtime_call_t`). Borrows its session, so the session cannot
/// change modules while it exists.
pub struct Function<'s> {
    call: sys::rocket_runtime_call_t,
    _session: PhantomData<&'s Session>,
}

impl Function<'_> {
    /// Invokes the function synchronously with `inputs`, in order, and
    /// returns its results. Every result must be a tensor.
    pub fn invoke(&mut self, inputs: &[&BufferView]) -> Result<Vec<BufferView>> {
        // SAFETY: the call was initialized against a session that outlives it
        // ('s); inputs are retained by the list for the invocation.
        unsafe {
            sys::rocket_runtime_call_reset(&mut self.call);
            for input in inputs {
                check(sys::rocket_runtime_call_inputs_push_back_buffer_view(
                    &mut self.call,
                    input.as_raw(),
                ))?;
            }
            check(sys::rocket_runtime_call_invoke(&mut self.call, 0))?;
            let count = sys::iree_vm_list_size(sys::rocket_runtime_call_outputs(&self.call));
            let mut outputs = Vec::with_capacity(count);
            for _ in 0..count {
                let mut raw = std::ptr::null_mut();
                check(sys::rocket_runtime_call_outputs_pop_front_buffer_view(
                    &mut self.call,
                    &mut raw,
                ))?;
                outputs.push(BufferView::from_raw(raw));
            }
            Ok(outputs)
        }
    }
}

impl Drop for Function<'_> {
    fn drop(&mut self) {
        // SAFETY: initialized in Session::function and not yet deinitialized.
        unsafe { sys::rocket_runtime_call_deinitialize(&mut self.call) };
    }
}

/// A module's bytecode, page-aligned the way IREE's own file loader leaves it.
struct AlignedBytes {
    ptr: NonNull<u8>,
    layout: Layout,
}

impl AlignedBytes {
    const ALIGNMENT: usize = 4096;

    fn copy_of(bytes: &[u8]) -> Self {
        let layout = Layout::from_size_align(bytes.len().max(1), Self::ALIGNMENT)
            .expect("module size overflows a Layout");
        // SAFETY: the layout is non-zero-sized.
        let ptr = NonNull::new(unsafe { std::alloc::alloc(layout) })
            .unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        // SAFETY: freshly allocated, at least bytes.len() long, not overlapping.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.as_ptr(), bytes.len()) };
        Self {
            ptr,
            layout: Layout::from_size_align(bytes.len(), Self::ALIGNMENT).unwrap(),
        }
    }
}

impl Drop for AlignedBytes {
    fn drop(&mut self) {
        let layout =
            Layout::from_size_align(self.layout.size().max(1), self.layout.align()).unwrap();
        // SAFETY: allocated in copy_of with this layout.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), layout) };
    }
}

/// The deallocator handed to IREE with a module's bytes: an
/// `iree_allocator_ctl_fn_t` whose only job is FREE, and whose `self` is the
/// boxed AlignedBytes.
unsafe extern "C" fn free_aligned_bytes(
    self_: *mut c_void,
    command: sys::iree_allocator_command_t,
    _params: *const c_void,
    _inout_ptr: *mut *mut c_void,
) -> sys::iree_status_t {
    if command as u32 == sys::iree_allocator_command_e_IREE_ALLOCATOR_COMMAND_FREE as u32 {
        // SAFETY: self_ is the Box made in load_module_bytes, freed exactly
        // once: IREE frees a module's flatbuffer once.
        drop(unsafe { Box::from_raw(self_.cast::<AlignedBytes>()) });
        std::ptr::null_mut()
    } else {
        sys::iree_status_from_code(sys::iree_status_code_e_IREE_STATUS_UNIMPLEMENTED as _)
    }
}
