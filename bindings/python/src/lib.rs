#![deny(missing_docs)]
//! Python bindings for the safetensors library.
use core::slice;
use dlpark::ffi as dlpack_ffi;
use dlpark::traits::{RowMajorCompactLayout, TensorLike};
use dlpark::SafeManagedTensor;
use pyo3::exceptions::{PyException, PyFileNotFoundError};
use pyo3::prelude::*;
use pyo3::sync::OnceLockExt;
use pyo3::types::IntoPyDict;
use pyo3::types::{PyBool, PyByteArray, PyBytes, PyDict, PyList, PySlice};
use pyo3::Bound as PyBound;
use pyo3::{intern, PyErr};
use safetensors::index::SafetensorsIndex;
use safetensors::loader::{
    resolve_prefix_map, Backend as LoaderBackend, Buffer as LoaderBuffer, BufferRef,
    Device as LoaderDevice, DeviceMap, FileLoader, Loader as ModelLoader, LoaderBuilder,
    TensorReady,
};
use safetensors::slice::TensorIndexer;
use safetensors::tensor::{Dtype, Metadata, SafeTensors, TensorInfo, TensorView};
use safetensors::View;
use std::borrow::Cow;
use std::collections::HashMap;
use std::ffi::CStr;
use std::ops::Bound;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::OnceLock;

// DLPack capsule names
const DLTENSOR: &CStr = c"dltensor";
const USED_DLTENSOR: &CStr = c"used_dltensor";

/// Capsule deleter for DLPack tensors.
/// Called when the PyCapsule is garbage collected.
unsafe extern "C" fn dlpack_capsule_deleter(capsule: *mut pyo3::ffi::PyObject) {
    unsafe {
        // If the capsule name is "used_dltensor", the tensor took ownership
        if pyo3::ffi::PyCapsule_IsValid(capsule, USED_DLTENSOR.as_ptr()) == 1 {
            return;
        }
        let ptr = pyo3::ffi::PyCapsule_GetPointer(capsule, DLTENSOR.as_ptr());
        if ptr.is_null() {
            pyo3::ffi::PyErr_WriteUnraisable(capsule);
            return;
        }
        // Reconstruct and drop the SafeManagedTensor
        let _ = SafeManagedTensor::from_raw(ptr as *mut dlpack_ffi::ManagedTensor);
    }
}

/// Convert SafeManagedTensor to PyCapsule.
fn managed_tensor_to_capsule(
    py: Python<'_>,
    managed_tensor: SafeManagedTensor,
) -> PyResult<PyObject> {
    unsafe {
        let raw_ptr = managed_tensor.into_raw();
        let capsule = pyo3::ffi::PyCapsule_New(
            raw_ptr as *mut std::ffi::c_void,
            DLTENSOR.as_ptr(),
            Some(dlpack_capsule_deleter),
        );
        if capsule.is_null() {
            return Err(PyErr::fetch(py));
        }
        Ok(PyObject::from_owned_ptr(py, capsule))
    }
}

static TORCH_MODULE: OnceLock<Py<PyModule>> = OnceLock::new();
static TORCH_FROM_DLPACK: OnceLock<Py<PyAny>> = OnceLock::new();
static NUMPY_MODULE: OnceLock<Py<PyModule>> = OnceLock::new();
static NUMPY_FROM_DLPACK: OnceLock<Py<PyAny>> = OnceLock::new();
static TENSORFLOW_MODULE: OnceLock<Py<PyModule>> = OnceLock::new();
static FLAX_MODULE: OnceLock<Py<PyModule>> = OnceLock::new();
static JAX_FROM_DLPACK: OnceLock<Py<PyAny>> = OnceLock::new();
static MLX_MODULE: OnceLock<Py<PyModule>> = OnceLock::new();
static MLX_FROM_DLPACK: OnceLock<Py<PyAny>> = OnceLock::new();
static PADDLE_MODULE: OnceLock<Py<PyModule>> = OnceLock::new();
static PADDLE_FROM_DLPACK: OnceLock<Py<PyAny>> = OnceLock::new();

/// A CPU buffer wrapper that exposes DLPack protocol for zero-copy tensor creation.
///
/// Implements `__dlpack__` and `__dlpack_device__` for the DLPack protocol, enabling
/// zero-copy tensor sharing with PyTorch, NumPy, JAX, TensorFlow, MLX, and Paddle.
///
/// Always exposes data as uint8 - callers use view() + reshape() for dtype conversion.
/// This simplifies the code path and works uniformly for all dtypes including exotic ones.
#[pyclass]
struct CpuBuffer {
    /// The underlying buffer wrapped in Arc for shared ownership with DLPack.
    /// This allows the DLPack wrapper to keep the buffer alive independently.
    buffer: Arc<LoaderBuffer>,
}

impl CpuBuffer {
    /// Create a new CpuBuffer from a LoaderBuffer.
    fn new(buffer: LoaderBuffer) -> Self {
        Self {
            buffer: Arc::new(buffer),
        }
    }
}

#[pymethods]
impl CpuBuffer {
    /// Returns the DLPack device tuple (device_type, device_id).
    /// CPU device is type 1 (kDLCPU), device_id 0.
    fn __dlpack_device__(&self) -> (u32, i32) {
        (1, 0) // kDLCPU = 1
    }

    /// Returns a DLPack capsule for zero-copy tensor sharing.
    /// Exports as 1D uint8 array - callers use view() + reshape() for dtype conversion.
    ///
    /// The `stream` parameter is accepted for DLPack protocol compliance but ignored
    /// for CPU buffers (no synchronization needed).
    ///
    /// # Panics
    /// Panics if buffer is empty (caller should handle empty tensors separately).
    #[pyo3(signature = (*, stream=None))]
    fn __dlpack__(&self, py: Python<'_>, stream: Option<isize>) -> PyResult<PyObject> {
        let _ = stream; // CPU buffers don't need stream synchronization
        let nbytes = self.buffer.len();
        debug_assert!(
            !self.buffer.as_ptr().is_null() && nbytes > 0,
            "Empty buffers should be handled by caller"
        );

        // Clone the Arc to share ownership with the DLPack wrapper.
        // When numpy (or other frameworks) consumes the DLPack tensor,
        // this Arc keeps the underlying buffer memory alive until the
        // framework's tensor is garbage collected.
        let wrapper = DlpackCpuWrapper {
            ptr: self.buffer.as_ptr() as *mut std::ffi::c_void,
            nbytes,
            _buffer: Arc::clone(&self.buffer),
        };

        let managed_tensor = SafeManagedTensor::new(wrapper).map_err(|e| {
            SafetensorError::new_err(format!("Failed to create DLPack tensor: {:?}", e))
        })?;

        managed_tensor_to_capsule(py, managed_tensor)
    }
}

/// Wrapper for creating DLPack tensor from CPU memory.
/// Exposes data as 1D uint8 array for uniform handling of all dtypes.
/// Holds an Arc<LoaderBuffer> to keep the underlying memory alive when
/// the DLPack tensor is consumed by frameworks like numpy.
struct DlpackCpuWrapper {
    ptr: *mut std::ffi::c_void,
    nbytes: usize,
    /// Keeps the buffer memory alive. When this wrapper is destroyed
    /// (via DLPack deleter), the Arc refcount decrements, potentially
    /// freeing the underlying mmap or owned memory.
    _buffer: Arc<LoaderBuffer>,
}

impl TensorLike<RowMajorCompactLayout> for DlpackCpuWrapper {
    type Error = std::convert::Infallible;

    fn data_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }

    fn memory_layout(&self) -> RowMajorCompactLayout {
        RowMajorCompactLayout::new(vec![self.nbytes as i64])
    }

    fn device(&self) -> Result<dlpack_ffi::Device, Self::Error> {
        Ok(dlpack_ffi::Device::CPU)
    }

    fn data_type(&self) -> Result<dlpack_ffi::DataType, Self::Error> {
        Ok(dlpack_ffi::DataType::U8)
    }

    fn byte_offset(&self) -> u64 {
        0
    }
}

/// Reference to GPU memory backing a tensor.
enum GpuBufferRef {
    /// Owned buffer from a per-tensor fetch — freed when this holder drops.
    Owned(LoaderBuffer),
    /// Shared reference to a preloaded bulk buffer — keeps it alive via Arc.
    Shared(Arc<LoaderBuffer>),
}

/// Lightweight holder to keep GPU buffer alive when attached to a tensor.
/// Attached via tensor._safetensors_buffer to prevent deallocation.
#[pyclass]
struct GpuBufferHolder {
    #[allow(dead_code)]
    buffer: GpuBufferRef,
}

/// Wrapper for creating DLPack tensor from GPU memory.
/// This struct owns all its data to avoid lifetime issues with PyO3.
/// The `_buffer` field keeps GPU memory alive through the DLPack deleter —
/// when PyTorch frees the tensor, it drops this wrapper, which drops the buffer.
/// This is essential because Python attributes (like `_safetensors_buffer`) are
/// lost when tensors are wrapped in `nn.Parameter` via `load_state_dict(assign=True)`.
struct DlpackGpuWrapper {
    ptr: *mut std::ffi::c_void,
    shape: Vec<i64>,
    dtype: dlpack_ffi::DataType,
    device_index: usize,
    /// Keeps GPU memory alive as long as the DLPack tensor exists.
    #[allow(dead_code)]
    _buffer: GpuBufferRef,
}

impl TensorLike<RowMajorCompactLayout> for DlpackGpuWrapper {
    type Error = std::convert::Infallible;

    fn data_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }

    fn memory_layout(&self) -> RowMajorCompactLayout {
        RowMajorCompactLayout::new(self.shape.clone())
    }

    fn device(&self) -> Result<dlpack_ffi::Device, Self::Error> {
        Ok(dlpack_ffi::Device::cuda(self.device_index))
    }

    fn data_type(&self) -> Result<dlpack_ffi::DataType, Self::Error> {
        Ok(self.dtype)
    }

    fn byte_offset(&self) -> u64 {
        0
    }
}

/// Convert safetensors Dtype to DLPack DataType.
/// Returns None for exotic types (F8, F4) which are exported as uint8 + view-cast.
fn dtype_to_dlpack(dtype: Dtype) -> Option<dlpack_ffi::DataType> {
    Some(match dtype {
        Dtype::BOOL => dlpack_ffi::DataType::BOOL,
        Dtype::U8 => dlpack_ffi::DataType::U8,
        Dtype::I8 => dlpack_ffi::DataType::I8,
        Dtype::U16 => dlpack_ffi::DataType::U16,
        Dtype::I16 => dlpack_ffi::DataType::I16,
        Dtype::U32 => dlpack_ffi::DataType::U32,
        Dtype::I32 => dlpack_ffi::DataType::I32,
        Dtype::U64 => dlpack_ffi::DataType::U64,
        Dtype::I64 => dlpack_ffi::DataType::I64,
        Dtype::F16 => dlpack_ffi::DataType::F16,
        Dtype::BF16 => dlpack_ffi::DataType::BF16,
        Dtype::F32 => dlpack_ffi::DataType::F32,
        Dtype::F64 => dlpack_ffi::DataType::F64,
        _ => return None,
    })
}

/// Create a framework tensor from a GPU buffer via DLPack.
///
/// Core shared helper for all CUDA tensor creation paths. Takes a GpuBufferRef
/// (Owned or Shared), creates a DLPack capsule, and dispatches to the framework's
/// `from_dlpack`. Handles exotic dtypes (F8, F4) by exporting as uint8 and view-casting.
#[allow(clippy::too_many_arguments)]
fn buffer_to_cuda_tensor(
    py: Python<'_>,
    buffer: GpuBufferRef,
    data_offset: usize,
    nbytes: usize,
    dtype: Dtype,
    shape: &[usize],
    framework: &Framework,
    device_index: usize,
) -> PyResult<PyObject> {
    let ptr = unsafe {
        match &buffer {
            GpuBufferRef::Owned(b) => {
                (b.as_ptr() as *mut u8).add(data_offset) as *mut std::ffi::c_void
            }
            GpuBufferRef::Shared(b) => {
                (b.as_ptr() as *mut u8).add(data_offset) as *mut std::ffi::c_void
            }
        }
    };

    let (dlpack_dtype, dlpack_shape) = if let Some(dt) = dtype_to_dlpack(dtype) {
        let shape_i64: Vec<i64> = shape.iter().map(|&s| s as i64).collect();
        (dt, shape_i64)
    } else {
        (dlpack_ffi::DataType::U8, vec![nbytes as i64])
    };

    let wrapper = DlpackGpuWrapper {
        ptr,
        shape: dlpack_shape,
        dtype: dlpack_dtype,
        device_index,
        _buffer: buffer,
    };

    let managed_tensor = SafeManagedTensor::new(wrapper).map_err(|e| {
        SafetensorError::new_err(format!("Failed to create DLPack tensor: {:?}", e))
    })?;

    let capsule = managed_tensor_to_capsule(py, managed_tensor)?;

    let tensor: PyBound<'_, PyAny> = match framework {
        Framework::Pytorch => {
            let from_dlpack = TORCH_FROM_DLPACK
                .get()
                .ok_or_else(|| SafetensorError::new_err("torch.from_dlpack not initialized"))?
                .bind(py);
            from_dlpack.call1((capsule,))?
        }
        Framework::Flax => {
            let from_dlpack = JAX_FROM_DLPACK
                .get()
                .ok_or_else(|| SafetensorError::new_err("jax.numpy.from_dlpack not initialized"))?
                .bind(py);
            from_dlpack.call1((capsule,))?
        }
        Framework::Paddle => {
            let from_dlpack = PADDLE_FROM_DLPACK
                .get()
                .ok_or_else(|| SafetensorError::new_err("paddle.from_dlpack not initialized"))?
                .bind(py);
            from_dlpack.call1((capsule,))?
        }
        _ => {
            return Err(SafetensorError::new_err(format!(
                "CUDA direct loading not supported for framework: {framework}"
            )));
        }
    };

    // For exotic dtypes, view-cast from uint8 to target dtype and reshape
    let tensor = if dtype_to_dlpack(dtype).is_none() {
        match framework {
            Framework::Pytorch => {
                let torch = get_module(py, &TORCH_MODULE)?;
                let target_dtype = get_pydtype(torch, dtype, false)?;
                let typed = tensor.call_method1(intern!(py, "view"), (target_dtype,))?;
                typed.call_method1(intern!(py, "reshape"), (shape.to_vec(),))?
            }
            _ => {
                return Err(SafetensorError::new_err(format!(
                    "Exotic dtype {dtype:?} not supported in {framework}"
                )));
            }
        }
    } else {
        tensor
    };

    Ok(tensor.into())
}

/// Create a framework tensor from a CPU buffer via DLPack or PyByteArray fallback.
///
/// Core shared helper for all CPU tensor creation paths. Handles empty tensors,
/// DLPack zero-copy for all frameworks, dtype view-casting, device transfer.
fn buffer_to_cpu_tensor(
    py: Python<'_>,
    buffer: LoaderBuffer,
    dtype: Dtype,
    shape: &[usize],
    framework: &Framework,
    device: &Device,
) -> PyResult<PyObject> {
    let nbytes = buffer.len();

    // Handle empty tensors (DLPack doesn't support null pointers)
    if nbytes == 0 || buffer.as_ptr().is_null() {
        let bytes = PyByteArray::new(py, &[]);
        return create_tensor(framework, dtype, shape, bytes.unbind().into_any(), device);
    }

    let cpu_buffer = CpuBuffer::new(buffer);
    let py_buffer = Py::new(py, cpu_buffer)?;
    let bound_buffer = py_buffer.bind(py);

    // Create tensor from DLPack protocol — pass the CpuBuffer object (which has
    // __dlpack__ and __dlpack_device__) directly to from_dlpack, NOT the raw capsule.
    let tensor: PyBound<'_, PyAny> = match framework {
        Framework::Pytorch => {
            let from_dlpack = TORCH_FROM_DLPACK
                .get()
                .ok_or_else(|| SafetensorError::new_err("torch.from_dlpack not initialized"))?
                .bind(py);
            from_dlpack.call1((bound_buffer,))?
        }
        Framework::Numpy => {
            let from_dlpack = NUMPY_FROM_DLPACK
                .get()
                .ok_or_else(|| SafetensorError::new_err("numpy.from_dlpack not initialized"))?
                .bind(py);
            from_dlpack.call1((bound_buffer,))?
        }
        Framework::Flax => {
            let from_dlpack = JAX_FROM_DLPACK
                .get()
                .ok_or_else(|| SafetensorError::new_err("jax.numpy.from_dlpack not initialized"))?
                .bind(py);
            from_dlpack.call1((bound_buffer,))?
        }
        Framework::Mlx => {
            let from_dlpack = MLX_FROM_DLPACK
                .get()
                .ok_or_else(|| SafetensorError::new_err("mlx.core.from_dlpack not initialized"))?
                .bind(py);
            from_dlpack.call1((bound_buffer,))?
        }
        Framework::Paddle => {
            let from_dlpack = PADDLE_FROM_DLPACK
                .get()
                .ok_or_else(|| SafetensorError::new_err("paddle.from_dlpack not initialized"))?
                .bind(py);
            from_dlpack.call1((bound_buffer,))?
        }
        Framework::Tensorflow => {
            // TensorFlow: use numpy as intermediate, then tf.constant (copies data)
            let np_from_dlpack = NUMPY_FROM_DLPACK
                .get()
                .ok_or_else(|| SafetensorError::new_err("numpy.from_dlpack not initialized"))?
                .bind(py);
            let uint8_array = np_from_dlpack.call1((bound_buffer,))?;
            let numpy = get_module(py, &NUMPY_MODULE)?;
            let np_dtype = get_pydtype(numpy, dtype, true)?;
            let typed_array = uint8_array.call_method1(intern!(py, "view"), (np_dtype,))?;
            let shaped_array =
                typed_array.call_method1(intern!(py, "reshape"), (shape.to_vec(),))?;
            let tf = TENSORFLOW_MODULE
                .get()
                .ok_or_else(|| SafetensorError::new_err("tensorflow module not initialized"))?
                .bind(py);
            return Ok(tf
                .getattr(intern!(py, "constant"))?
                .call1((shaped_array,))?
                .into());
        }
    };

    // View as proper dtype (from u8) and reshape
    let tensor = match framework {
        Framework::Pytorch => {
            let torch = get_module(py, &TORCH_MODULE)?;
            let target_dtype = get_pydtype(torch, dtype, false)?;
            let typed = tensor.call_method1(intern!(py, "view"), (target_dtype,))?;
            typed.call_method1(intern!(py, "reshape"), (shape.to_vec(),))?
        }
        Framework::Numpy => {
            let np = get_module(py, &NUMPY_MODULE)?;
            let target_dtype = get_pydtype(np, dtype, true)?;
            tensor
                .call_method1(intern!(py, "view"), (target_dtype,))?
                .call_method1(intern!(py, "reshape"), (shape.to_vec(),))?
        }
        Framework::Flax => {
            let jax = get_module(py, &FLAX_MODULE)?;
            let target_dtype = get_jax_dtype(jax, dtype)?;
            tensor
                .call_method1(intern!(py, "view"), (target_dtype,))?
                .call_method1(intern!(py, "reshape"), (shape.to_vec(),))?
        }
        Framework::Mlx => {
            let mlx = get_module(py, &MLX_MODULE)?;
            let target_dtype = get_mlx_dtype(mlx, dtype)?;
            tensor
                .call_method1(intern!(py, "view"), (target_dtype,))?
                .call_method1(intern!(py, "reshape"), (shape.to_vec(),))?
        }
        Framework::Paddle => {
            let paddle = get_module(py, &PADDLE_MODULE)?;
            let target_dtype = get_paddle_dtype(paddle, dtype)?;
            let casted = paddle.call_method1(intern!(py, "cast"), (tensor, target_dtype))?;
            paddle.call_method1(intern!(py, "reshape"), (casted, shape.to_vec()))?
        }
        Framework::Tensorflow => unreachable!(), // handled above via early return
    };

    // Keep buffer alive — only for frameworks that support arbitrary attributes.
    // Numpy/JAX/MLX keep the reference via DLPack mechanism internally.
    if matches!(framework, Framework::Pytorch | Framework::Paddle) {
        tensor.setattr(intern!(py, "_safetensors_buffer"), py_buffer)?;
    }

    // Move tensor to target device if needed
    let tensor = if *device != Device::Cpu {
        match framework {
            Framework::Pytorch | Framework::Paddle => {
                let device_str = format!("{device}");
                tensor.call_method1(intern!(py, "to"), (device_str,))?
            }
            _ => tensor,
        }
    } else {
        tensor
    };

    Ok(tensor.into())
}

/// Fetch a range of bytes and create a CUDA tensor with ZERO COPY.
#[allow(clippy::too_many_arguments)]
fn fetch_cuda_tensor(
    loader: &FileLoader,
    py: Python<'_>,
    start: usize,
    end: usize,
    dtype: Dtype,
    shape: &[usize],
    framework: &Framework,
    device_index: usize,
) -> PyResult<PyObject> {
    let buffer = loader
        .fetch(start, end)
        .map_err(|e| SafetensorError::new_err(format!("Loader fetch error: {e}")))?;
    let nbytes = buffer.len();
    buffer_to_cuda_tensor(
        py,
        GpuBufferRef::Owned(buffer),
        0,
        nbytes,
        dtype,
        shape,
        framework,
        device_index,
    )
}

/// Create a CUDA tensor from a pointer into a shared preloaded GPU buffer.
#[allow(clippy::too_many_arguments)]
fn cuda_tensor_from_preloaded(
    py: Python<'_>,
    preloaded: &Arc<LoaderBuffer>,
    data_offset: usize,
    nbytes: usize,
    dtype: Dtype,
    shape: &[usize],
    framework: &Framework,
    device_index: usize,
) -> PyResult<PyObject> {
    buffer_to_cuda_tensor(
        py,
        GpuBufferRef::Shared(preloaded.clone()),
        data_offset,
        nbytes,
        dtype,
        shape,
        framework,
        device_index,
    )
}

/// Fetch a range of bytes and create a CPU tensor using zero-copy via DLPack.
#[allow(clippy::too_many_arguments)]
fn fetch_cpu_tensor(
    loader: &FileLoader,
    py: Python<'_>,
    start: usize,
    end: usize,
    dtype: Dtype,
    shape: &[usize],
    framework: &Framework,
    device: &Device,
) -> PyResult<PyObject> {
    // Try zero-copy mmap view first (only works for CPU loaders).
    let buffer = match loader.fetch_view(start, end) {
        Ok(buf) => buf,
        Err(_) => {
            // Loader is CUDA or fetch_view unsupported — copy to CPU via fetch_to_vec
            let vec = loader
                .fetch_to_vec(start, end)
                .map_err(|e| SafetensorError::new_err(format!("Loader fetch error: {e}")))?;
            let array: PyObject = PyByteArray::new(py, &vec).into_any().into();
            return create_tensor(framework, dtype, shape, array, device);
        }
    };

    buffer_to_cpu_tensor(py, buffer, dtype, shape, framework, device)
}

struct TensorDataPointer {
    addr: u64,
    len: usize,
}

struct TensorRawDataView {
    shape: Vec<usize>,
    dtype: Dtype,
    tensor_data_ptr: TensorDataPointer,
}

impl View for &TensorRawDataView {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        let p = self.tensor_data_ptr.addr as *const u8;
        unsafe {
            let slice = slice::from_raw_parts(p, self.tensor_data_ptr.len);
            Cow::Borrowed(slice)
        }
    }

    fn data_len(&self) -> usize {
        self.tensor_data_ptr.len
    }
}

fn prepare_shape(tensor_desc: &PyBound<PyDict>) -> PyResult<Vec<usize>> {
    tensor_desc
        .get_item("shape")?
        .ok_or_else(|| SafetensorError::new_err(format!("Missing `shape` in {tensor_desc}")))?
        .extract()
}

fn prepare_dtype(tensor_desc: &PyBound<PyDict>) -> PyResult<Dtype> {
    let pydtype = tensor_desc
        .get_item("dtype")?
        .ok_or_else(|| SafetensorError::new_err(format!("Missing `dtype` in {tensor_desc}")))?;
    let dtype: String = pydtype.extract()?;
    let dtype = match dtype.as_ref() {
        "bool" => Dtype::BOOL,
        "int8" => Dtype::I8,
        "uint8" => Dtype::U8,
        "int16" => Dtype::I16,
        "uint16" => Dtype::U16,
        "int32" => Dtype::I32,
        "uint32" => Dtype::U32,
        "int64" => Dtype::I64,
        "uint64" => Dtype::U64,
        "float16" => Dtype::F16,
        "float32" => Dtype::F32,
        "float64" => Dtype::F64,
        "bfloat16" => Dtype::BF16,
        "float8_e4m3fn" => Dtype::F8_E4M3,
        "float8_e5m2" => Dtype::F8_E5M2,
        "float8_e8m0fnu" => Dtype::F8_E8M0,
        "float4_e2m1fn_x2" => Dtype::F4,
        "complex64" => Dtype::C64,
        dtype_str => {
            return Err(SafetensorError::new_err(format!(
                "dtype {dtype_str} is not covered",
            )));
        }
    };
    Ok(dtype)
}

fn prepare_tensor_raw_data_view(
    tensor_dict: HashMap<String, PyBound<PyDict>>,
) -> PyResult<HashMap<String, TensorRawDataView>> {
    let mut tensors = HashMap::with_capacity(tensor_dict.len());
    for (tensor_name, tensor_desc) in tensor_dict {
        let data_ptr: u64 = tensor_desc
            .get_item("data_ptr")?
            .ok_or_else(|| {
                SafetensorError::new_err(format!("Missing `data_ptr` in {tensor_desc}"))
            })?
            .extract()?;
        let data_len: usize = tensor_desc
            .get_item("data_len")?
            .ok_or_else(|| {
                SafetensorError::new_err(format!("Missing `data_len` in {tensor_desc}"))
            })?
            .extract()?;
        let dtype = prepare_dtype(&tensor_desc)?;
        let mut shape = prepare_shape(&tensor_desc)?;
        if dtype == Dtype::F4 {
            let n = shape.len();
            shape[n - 1] *= 2;
        }
        let tensor_data_ptr = TensorDataPointer {
            addr: data_ptr,
            len: data_len,
        };
        let tensor = TensorRawDataView {
            shape,
            dtype,
            tensor_data_ptr,
        };
        tensors.insert(tensor_name, tensor);
    }
    Ok(tensors)
}

/// Serializes raw data.
///
/// NOTE: the caller is required to ensure any pointer passed via `data_ptr` is valid and will live
/// long enough for the duration of the serialization.
/// We will remove the need for the caller to hold references themselves when we drop support for
/// python versions prior to 3.11 where the `PyBuffer` API is available.
/// Creating a `PyBuffer` will enable us to hold a reference to each passed in data array,
/// increasing its ref count preventing the gc from collecting it while we serialize.
///
/// Args:
///     tensor_dict (`Dict[str, Dict[Any]]`):
///         The tensor dict is like:
///             {"tensor_name": {"dtype": "F32", "shape": [2, 3], "data_ptr": 1234, "data_len": 24}}
///     metadata (`Dict[str, str]`, *optional*):
///         The optional purely text annotations
///
/// Returns:
///     (`bytes`):
///         The serialized content.
#[pyfunction]
#[pyo3(signature = (tensor_dict, metadata=None))]
fn serialize<'b>(
    py: Python<'b>,
    tensor_dict: HashMap<String, PyBound<PyDict>>,
    metadata: Option<HashMap<String, String>>,
) -> PyResult<PyBound<'b, PyBytes>> {
    let tensors = prepare_tensor_raw_data_view(tensor_dict)?;
    let out = py
        .allow_threads(|| safetensors::tensor::serialize(&tensors, metadata))
        .map_err(|e| SafetensorError::new_err(format!("Error while serializing: {e}")))?;
    let pybytes = PyBytes::new(py, &out);
    Ok(pybytes)
}

/// Serializes raw data into file.
///
/// NOTE: the caller is required to ensure any pointer passed via `data_ptr` is valid and will live
/// long enough for the duration of the serialization.
/// We will remove the need for the caller to hold references themselves when we drop support for
/// python versions prior to 3.11 where the `PyBuffer` API is available.
/// Creating a `PyBuffer` will enable us to hold a reference to each passed in data array,
/// increasing its ref count preventing the gc from collecting it while we serialize.
///
/// Args:
///     tensor_dict (`Dict[str, Dict[Any]]`):
///         The tensor dict is like:
///             {"tensor_name": {"dtype": "F32", "shape": [2, 3], "data_ptr": 1234, "data_len": 24}}
///     filename (`str`, or `os.PathLike`):
///         The name of the file to write into.
///     metadata (`Dict[str, str]`, *optional*):
///         The optional purely text annotations
///
/// Returns:
///     (`NoneType`):
///         On success return None
#[pyfunction]
#[pyo3(signature = (tensor_dict, filename, metadata=None))]
fn serialize_file(
    py: Python<'_>,
    tensor_dict: HashMap<String, PyBound<PyDict>>,
    filename: PathBuf,
    metadata: Option<HashMap<String, String>>,
) -> PyResult<()> {
    let tensors = prepare_tensor_raw_data_view(tensor_dict)?;
    py.allow_threads(|| {
        safetensors::tensor::serialize_to_file(&tensors, metadata, filename.as_path())
            .map_err(|e| SafetensorError::new_err(format!("Error while serializing: {e}")))
    })?;

    Ok(())
}

/// Opens a safetensors lazily and returns tensors as asked
///
/// Args:
///     data (`bytes`):
///         The byte content of a file
///
/// Returns:
///     (`List[str, Dict[str, Dict[str, any]]]`):
///         The deserialized content is like:
///             [("tensor_name", {"shape": [2, 3], "dtype": "F32", "data": b"\0\0.." }), (...)]
#[pyfunction]
#[pyo3(signature = (bytes))]
fn deserialize(py: Python, bytes: &[u8]) -> PyResult<Vec<(String, HashMap<String, PyObject>)>> {
    let safetensor = SafeTensors::deserialize(bytes)
        .map_err(|e| SafetensorError::new_err(format!("Error while deserializing: {e}")))?;

    let tensors = safetensor.tensors();
    let mut items = Vec::with_capacity(tensors.len());

    for (tensor_name, tensor) in tensors {
        let pyshape: PyObject = PyList::new(py, tensor.shape().iter())?.into();
        let pydtype: PyObject = tensor.dtype().to_string().into_pyobject(py)?.into();

        let pydata: PyObject = PyByteArray::new(py, tensor.data()).into();

        let map = HashMap::from([
            ("shape".to_string(), pyshape),
            ("dtype".to_string(), pydtype),
            ("data".to_string(), pydata),
        ]);
        items.push((tensor_name, map));
    }
    Ok(items)
}

fn slice_to_indexer(
    (dim_idx, (slice_index, dim)): (usize, (SliceIndex, usize)),
) -> Result<TensorIndexer, PyErr> {
    match slice_index {
        SliceIndex::Slice(slice) => {
            let py_start = slice.getattr(intern!(slice.py(), "start"))?;
            let start: Option<usize> = py_start.extract()?;
            let start = if let Some(start) = start {
                Bound::Included(start)
            } else {
                Bound::Unbounded
            };

            let py_stop = slice.getattr(intern!(slice.py(), "stop"))?;
            let stop: Option<usize> = py_stop.extract()?;
            let stop = if let Some(stop) = stop {
                Bound::Excluded(stop)
            } else {
                Bound::Unbounded
            };
            Ok(TensorIndexer::Narrow(start, stop))
        }
        SliceIndex::Index(idx) => {
            if idx < 0 {
                let idx = dim.checked_add_signed(idx as isize).ok_or_else(|| {
                    SafetensorError::new_err(format!(
                        "Invalid index {idx} for dimension {dim_idx} of size {dim}"
                    ))
                })?;
                Ok(TensorIndexer::Select(idx))
            } else {
                Ok(TensorIndexer::Select(idx as usize))
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Framework {
    Pytorch,
    Numpy,
    Tensorflow,
    Flax,
    Mlx,
    Paddle,
}

impl fmt::Display for Framework {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match *self {
            Framework::Pytorch => "pytorch",
            Framework::Numpy => "numpy",
            Framework::Tensorflow => "tensorflow",
            Framework::Flax => "flax",
            Framework::Mlx => "mlx",
            Framework::Paddle => "paddle",
        })
    }
}

impl<'source> FromPyObject<'source> for Framework {
    fn extract_bound(ob: &PyBound<'source, PyAny>) -> PyResult<Self> {
        let name: String = ob.extract()?;
        match &name[..] {
            "pt" => Ok(Framework::Pytorch),
            "torch" => Ok(Framework::Pytorch),
            "pytorch" => Ok(Framework::Pytorch),

            "np" => Ok(Framework::Numpy),
            "numpy" => Ok(Framework::Numpy),

            "tf" => Ok(Framework::Tensorflow),
            "tensorflow" => Ok(Framework::Tensorflow),

            "jax" => Ok(Framework::Flax),
            "flax" => Ok(Framework::Flax),
            "mlx" => Ok(Framework::Mlx),

            "paddle" => Ok(Framework::Paddle),
            name => Err(SafetensorError::new_err(format!(
                "framework {name} is invalid"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Device {
    Cpu,
    Cuda(usize),
    Mps,
    Npu(usize),
    Xpu(usize),
    Xla(usize),
    Mlu(usize),
    Musa(usize),
    Hpu(usize),
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Device::Cpu => write!(f, "cpu"),
            Device::Mps => write!(f, "mps"),
            Device::Cuda(index) => write!(f, "cuda:{index}"),
            Device::Musa(index) => write!(f, "musa:{index}"),
            Device::Npu(index) => write!(f, "npu:{index}"),
            Device::Xpu(index) => write!(f, "xpu:{index}"),
            Device::Xla(index) => write!(f, "xla:{index}"),
            Device::Mlu(index) => write!(f, "mlu:{index}"),
            Device::Hpu(index) => write!(f, "hpu:{index}"),
        }
    }
}

/// Parsing the device index.
fn parse_device(name: &str) -> PyResult<usize> {
    let tokens: Vec<_> = name.split(':').collect();
    if tokens.len() == 2 {
        Ok(tokens[1].parse()?)
    } else {
        Err(SafetensorError::new_err(format!(
            "device {name} is invalid"
        )))
    }
}

impl<'source> FromPyObject<'source> for Device {
    fn extract_bound(ob: &PyBound<'source, PyAny>) -> PyResult<Self> {
        if let Ok(name) = ob.extract::<String>() {
            match name.as_str() {
                "cpu" => Ok(Device::Cpu),
                "cuda" => Ok(Device::Cuda(0)),
                "musa" => Ok(Device::Musa(0)),
                "mps" => Ok(Device::Mps),
                "npu" => Ok(Device::Npu(0)),
                "xpu" => Ok(Device::Xpu(0)),
                "xla" => Ok(Device::Xla(0)),
                "mlu" => Ok(Device::Mlu(0)),
                "hpu" => Ok(Device::Hpu(0)),
                name if name.starts_with("cuda:") => parse_device(name).map(Device::Cuda),
                name if name.starts_with("musa:") => parse_device(name).map(Device::Musa),
                name if name.starts_with("npu:") => parse_device(name).map(Device::Npu),
                name if name.starts_with("xpu:") => parse_device(name).map(Device::Xpu),
                name if name.starts_with("xla:") => parse_device(name).map(Device::Xla),
                name if name.starts_with("mlu:") => parse_device(name).map(Device::Mlu),
                name if name.starts_with("hpu:") => parse_device(name).map(Device::Hpu),
                name => Err(SafetensorError::new_err(format!(
                    "device {name} is invalid"
                ))),
            }
        } else if let Ok(number) = ob.extract::<usize>() {
            Ok(Device::Cuda(number))
        } else {
            Err(SafetensorError::new_err(format!("device {ob} is invalid")))
        }
    }
}

impl<'py> IntoPyObject<'py> for Device {
    type Target = PyAny;
    type Output = pyo3::Bound<'py, Self::Target>;
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            Device::Cpu => "cpu".into_pyobject(py).map(|x| x.into_any()),
            Device::Cuda(n) => format!("cuda:{n}").into_pyobject(py).map(|x| x.into_any()),
            Device::Musa(n) => format!("musa:{n}").into_pyobject(py).map(|x| x.into_any()),
            Device::Mps => "mps".into_pyobject(py).map(|x| x.into_any()),
            Device::Npu(n) => format!("npu:{n}").into_pyobject(py).map(|x| x.into_any()),
            Device::Xpu(n) => format!("xpu:{n}").into_pyobject(py).map(|x| x.into_any()),
            Device::Xla(n) => format!("xla:{n}").into_pyobject(py).map(|x| x.into_any()),
            Device::Mlu(n) => format!("mlu:{n}").into_pyobject(py).map(|x| x.into_any()),
            Device::Hpu(n) => format!("hpu:{n}").into_pyobject(py).map(|x| x.into_any()),
        }
    }
}

/// Loader backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Backend {
    /// Automatically select the best backend for the platform.
    #[default]
    Auto,
    /// Use mmap-based loading (cross-platform).
    Mmap,
    /// Use io_uring for async I/O (Linux only).
    #[cfg(all(target_os = "linux", feature = "io_uring"))]
    IoUring,
    /// Use cuFile/GDS for direct NVMe->GPU DMA (Linux + CUDA only).
    #[cfg(all(target_os = "linux", feature = "cufile"))]
    CuFile,
}

impl Backend {
    /// Convert to loader Backend.
    fn to_loader(self) -> LoaderBackend {
        match self {
            Backend::Auto => LoaderBackend::Auto,
            Backend::Mmap => LoaderBackend::Mmap,
            #[cfg(all(target_os = "linux", feature = "io_uring"))]
            Backend::IoUring => LoaderBackend::IoUring,
            #[cfg(all(target_os = "linux", feature = "cufile"))]
            Backend::CuFile => LoaderBackend::CuFile,
        }
    }
}

impl<'source> FromPyObject<'source> for Backend {
    fn extract_bound(ob: &PyBound<'source, PyAny>) -> PyResult<Self> {
        let name: String = ob.extract()?;
        match name.as_str() {
            "auto" => Ok(Backend::Auto),
            "mmap" => Ok(Backend::Mmap),
            #[cfg(all(target_os = "linux", feature = "io_uring"))]
            "io_uring" | "iouring" => Ok(Backend::IoUring),
            #[cfg(not(all(target_os = "linux", feature = "io_uring")))]
            "io_uring" | "iouring" => Err(SafetensorError::new_err(
                "io_uring backend is only available on Linux with io_uring feature enabled",
            )),
            #[cfg(all(target_os = "linux", feature = "cufile"))]
            "cufile" | "gds" => Ok(Backend::CuFile),
            #[cfg(not(all(target_os = "linux", feature = "cufile")))]
            "cufile" | "gds" => Err(SafetensorError::new_err(
                "cuFile/GDS backend is only available on Linux with cufile feature enabled",
            )),
            name => Err(SafetensorError::new_err(format!(
                "backend {name} is invalid (valid: auto, mmap, io_uring, cufile, gds)"
            ))),
        }
    }
}

/// Device specification for safe_open.
///
/// Can be either:
/// - A single device (string like "cpu", "cuda:0", or int like 0)
/// - A dict mapping tensor names to devices for multi-GPU loading
#[derive(Debug, Clone)]
enum DeviceSpec {
    /// All tensors go to a single device.
    Single(Device),
    /// Prefix map from tensor names to devices.
    /// An empty-string key `""` acts as the catch-all default (resolved last
    /// by `resolve_prefix_map`). When absent, unmapped tensors go to CPU.
    Map(HashMap<String, Device>),
}

impl Default for DeviceSpec {
    fn default() -> Self {
        DeviceSpec::Single(Device::Cpu)
    }
}

impl DeviceSpec {
    /// Resolve which device a tensor should be loaded to.
    ///
    /// For `Map` variants, tries exact match first, then falls back to
    /// prefix matching (walking up the module tree). This allows transformers-style
    /// module-prefix device maps like `{"model.layers.0": "cuda:0"}` to work
    /// alongside exact tensor name maps.
    fn resolve(&self, tensor_name: &str) -> Device {
        match self {
            DeviceSpec::Single(device) => device.clone(),
            DeviceSpec::Map(map) => resolve_prefix_map(map, tensor_name)
                .cloned()
                .unwrap_or(Device::Cpu),
        }
    }

    /// Get the default/primary device (the `""` catch-all, or CPU if absent).
    fn default_device(&self) -> Device {
        match self {
            DeviceSpec::Single(device) => device.clone(),
            DeviceSpec::Map(map) => map.get("").cloned().unwrap_or(Device::Cpu),
        }
    }

    /// Check if this is a single-device spec (no multi-GPU routing).
    fn is_single(&self) -> bool {
        matches!(self, DeviceSpec::Single(_))
    }

    /// Get all unique devices used in this spec.
    fn devices(&self) -> Vec<Device> {
        match self {
            DeviceSpec::Single(device) => vec![device.clone()],
            DeviceSpec::Map(map) => {
                let mut devices: Vec<Device> = map.values().cloned().collect();
                devices.sort_by(|a, b| format!("{a}").cmp(&format!("{b}")));
                devices.dedup_by(|a, b| format!("{a}") == format!("{b}"));
                devices
            }
        }
    }
}

impl<'source> FromPyObject<'source> for DeviceSpec {
    fn extract_bound(ob: &PyBound<'source, PyAny>) -> PyResult<Self> {
        // Try to extract as a dict first (device map)
        if let Ok(dict) = ob.downcast::<pyo3::types::PyDict>() {
            let mut map = HashMap::new();
            for (key, value) in dict.iter() {
                let tensor_name: String = key.extract()?;
                let device: Device = value.extract()?;
                map.insert(tensor_name, device);
            }
            return Ok(DeviceSpec::Map(map));
        }

        // Otherwise, try single device
        let device: Device = ob.extract()?;
        Ok(DeviceSpec::Single(device))
    }
}

// ─── Tensor Parallelism (TP) Support ─────────────────────────────────────────

/// Tensor parallelism slicing strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
enum TpStrategy {
    /// Column-wise: slice dim -2 for 2D (output features), dim -1 for 1D (bias).
    /// Contiguous in row-major layout → single byte range.
    Colwise,
    /// Row-wise: slice dim -1 for 2D (input features), replicate for 1D (bias).
    /// Non-contiguous in row-major layout → load full tensor, narrow on GPU.
    Rowwise,
    /// Full copy to all ranks.
    Replicate,
}

/// Per-tensor TP metadata, precomputed at `safe_open()` time.
#[derive(Debug, Clone)]
struct TpTensorMeta {
    /// Byte range to load (relative to shard data section start).
    /// Colwise: the contiguous slice. Rowwise: full tensor range.
    load_offsets: (usize, usize),
    /// Shape of the tensor returned to the caller (after slicing).
    sliced_shape: Vec<usize>,
    /// For rowwise 2D: `Some((dim, start, length))` to narrow after creation.
    /// For everything else: `None`.
    narrow: Option<(usize, usize, usize)>,
}

/// Compiled TP plan: maps wildcard patterns to slicing strategies.
struct TpPlan {
    /// `(pattern, strategy)` pairs. Patterns use `*` to match digit sequences.
    patterns: Vec<(String, TpStrategy)>,
    rank: usize,
    world_size: usize,
}

impl TpPlan {
    /// Compile a TP plan from Python dict `{"layers.*.q_proj": "colwise", ...}`.
    fn compile(
        raw: &HashMap<String, String>,
        rank: usize,
        world_size: usize,
    ) -> Result<Self, String> {
        let mut patterns = Vec::with_capacity(raw.len());
        for (pattern, strategy_str) in raw {
            let strategy = match strategy_str.as_str() {
                "colwise" | "colwise_gather_output" | "colwise_rep" => TpStrategy::Colwise,
                "rowwise" | "rowwise_rep" | "rowwise_split_input" => TpStrategy::Rowwise,
                "replicate" | "sequence_parallel" => TpStrategy::Replicate,
                other => return Err(format!("Unknown TP strategy '{other}'")),
            };
            patterns.push((pattern.clone(), strategy));
        }
        Ok(Self {
            patterns,
            rank,
            world_size,
        })
    }

    /// Match a wildcard pattern against a tensor name.
    /// `*` in the pattern matches one or more ASCII digits in the name.
    /// Tensor names may have a trailing `.weight` or `.bias` suffix that is
    /// stripped before matching, mirroring transformers' module-level tp_plan keys.
    fn pattern_matches(pattern: &str, tensor_name: &str) -> bool {
        // Strip trailing .weight/.bias from tensor name for module-level matching
        let name = tensor_name
            .strip_suffix(".weight")
            .or_else(|| tensor_name.strip_suffix(".bias"))
            .unwrap_or(tensor_name);

        let parts: Vec<&str> = pattern.split('*').collect();
        if parts.len() == 1 {
            return pattern == name;
        }

        let mut remaining = name;
        for (i, part) in parts.iter().enumerate() {
            if i == 0 {
                if !remaining.starts_with(part) {
                    return false;
                }
                remaining = &remaining[part.len()..];
            } else {
                // Must consume one or more ASCII digits
                if remaining.is_empty() || !remaining.as_bytes()[0].is_ascii_digit() {
                    return false;
                }
                let digit_end = remaining
                    .bytes()
                    .position(|b| !b.is_ascii_digit())
                    .unwrap_or(remaining.len());
                remaining = &remaining[digit_end..];
                // Then match the literal part
                if !remaining.starts_with(part) {
                    return false;
                }
                remaining = &remaining[part.len()..];
            }
        }
        remaining.is_empty()
    }

    /// Resolve a tensor name to a TP strategy, or None if not in the plan.
    fn resolve(&self, tensor_name: &str) -> Option<TpStrategy> {
        for (pattern, strategy) in &self.patterns {
            if Self::pattern_matches(pattern, tensor_name) {
                return Some(*strategy);
            }
        }
        None
    }
}

/// Validate that the framework supports the given device/device_spec combination.
fn validate_device_framework(
    framework: &Framework,
    device_spec: &DeviceSpec,
    device: &Device,
) -> PyResult<()> {
    if !device_spec.is_single()
        && *framework != Framework::Pytorch
        && *framework != Framework::Paddle
    {
        return Err(SafetensorError::new_err(format!(
            "Device maps are only supported for PyTorch and Paddle frameworks, got {framework}",
        )));
    }
    if *device != Device::Cpu && *framework != Framework::Pytorch && *framework != Framework::Paddle
    {
        return Err(SafetensorError::new_err(format!(
            "Device {device} is not supported for framework {framework}",
        )));
    }
    Ok(())
}

/// Import framework modules and cache `from_dlpack` callables.
///
/// Sets the initial CUDA device for PyTorch when the device spec includes CUDA targets.
fn init_framework_modules(
    py: Python<'_>,
    framework: &Framework,
    device_spec: &DeviceSpec,
    device: &Device,
) -> PyResult<()> {
    let numpy = PyModule::import(py, intern!(py, "numpy"))?;
    let np_from_dlpack = numpy.getattr(intern!(py, "from_dlpack"))?;
    NUMPY_FROM_DLPACK.get_or_init_py_attached(py, || np_from_dlpack.unbind());
    NUMPY_MODULE.get_or_init_py_attached(py, || numpy.into());

    match *framework {
        Framework::Pytorch => {
            let module = PyModule::import(py, intern!(py, "torch"))?;
            let from_dlpack = module.getattr(intern!(py, "from_dlpack"))?;
            TORCH_FROM_DLPACK.get_or_init_py_attached(py, || from_dlpack.unbind());
            let init_cuda_idx = match *device {
                Device::Cuda(idx) => Some(idx),
                _ => device_spec.devices().iter().find_map(|d| match d {
                    Device::Cuda(idx) => Some(*idx),
                    _ => None,
                }),
            };
            if let Some(idx) = init_cuda_idx {
                module
                    .getattr(intern!(py, "cuda"))?
                    .getattr(intern!(py, "set_device"))?
                    .call1((idx,))?;
            }
            TORCH_MODULE.get_or_init_py_attached(py, || module.into());
        }
        Framework::Flax => {
            let jax = PyModule::import(py, intern!(py, "jax"))?;
            let from_dlpack = jax
                .getattr(intern!(py, "numpy"))?
                .getattr(intern!(py, "from_dlpack"))?;
            JAX_FROM_DLPACK.get_or_init_py_attached(py, || from_dlpack.unbind());
            FLAX_MODULE.get_or_init_py_attached(py, || jax.into());
        }
        Framework::Tensorflow => {
            let tf = PyModule::import(py, intern!(py, "tensorflow"))?;
            TENSORFLOW_MODULE.get_or_init_py_attached(py, || tf.into());
        }
        Framework::Mlx => {
            let mlx = PyModule::import(py, intern!(py, "mlx"))?;
            let from_dlpack = mlx
                .getattr(intern!(py, "core"))?
                .getattr(intern!(py, "from_dlpack"))?;
            MLX_FROM_DLPACK.get_or_init_py_attached(py, || from_dlpack.unbind());
            MLX_MODULE.get_or_init_py_attached(py, || mlx.into());
        }
        Framework::Paddle => {
            let paddle = PyModule::import(py, intern!(py, "paddle"))?;
            let from_dlpack = paddle.getattr(intern!(py, "from_dlpack"))?;
            PADDLE_FROM_DLPACK.get_or_init_py_attached(py, || from_dlpack.unbind());
            PADDLE_MODULE.get_or_init_py_attached(py, || paddle.into());
        }
        Framework::Numpy => {} // already initialized above
    };
    Ok(())
}

/// Determine the loader device based on framework and device spec.
///
/// Uses CUDA loader only when the framework supports it AND there are no CPU targets
/// Convert Python DeviceSpec to Rust DeviceMap.
///
/// Maps CUDA devices to LoaderDevice::Cuda for CUDA-capable frameworks;
/// all other devices (MPS, NPU, XPU, etc.) map to LoaderDevice::Cpu
/// (Python bindings handle the .to(device) transfer after loading).
fn device_spec_to_device_map(device_spec: &DeviceSpec, framework: &Framework) -> DeviceMap {
    let cuda_capable = matches!(
        framework,
        Framework::Pytorch | Framework::Flax | Framework::Paddle
    );

    match device_spec {
        DeviceSpec::Single(ref device) => {
            if cuda_capable {
                if let Device::Cuda(idx) = device {
                    return DeviceMap::single(LoaderDevice::Cuda(*idx));
                }
            }
            DeviceMap::single(LoaderDevice::Cpu)
        }
        DeviceSpec::Map(ref map) => {
            let mut loader_map = HashMap::new();
            for (name, device) in map {
                if cuda_capable {
                    if let Device::Cuda(idx) = device {
                        loader_map.insert(name.clone(), LoaderDevice::Cuda(*idx));
                        continue;
                    }
                }
                loader_map.insert(name.clone(), LoaderDevice::Cpu);
            }
            DeviceMap::from_map(loader_map, LoaderDevice::Cpu)
        }
    }
}

/// Adjust shape for F4 dtype in PyTorch (which packs 2 values per byte).
///
/// PyTorch exposes F4 tensors with halved last dimension. Returns the
/// original shape unchanged for non-F4 dtypes or non-PyTorch frameworks.
fn adjust_f4_shape(framework: &Framework, dtype: Dtype, shape: Vec<usize>) -> PyResult<Vec<usize>> {
    if *framework == Framework::Pytorch && dtype == Dtype::F4 {
        let mut shape = shape;
        let n = shape.len();
        if shape[n - 1] % 2 != 0 {
            return Err(SafetensorError::new_err(format!(
                "f4_x2 dtype requires that the last dim be divisible by 2 in torch: got {shape:?}",
            )));
        }
        shape[n - 1] /= 2;
        Ok(shape)
    } else {
        Ok(shape)
    }
}

/// Compute TP metadata for a single tensor.
///
/// Given the tensor's shape, dtype, file offsets, and the TP strategy/rank/world_size,
/// returns the byte range to load and the shape to expose to the caller.
fn compute_tp_tensor_meta(
    shape: &[usize],
    dtype: Dtype,
    data_offsets: (usize, usize),
    strategy: TpStrategy,
    rank: usize,
    world_size: usize,
) -> TpTensorMeta {
    let ndim = shape.len();
    let elem_bits = dtype.bitsize();
    let data_start = data_offsets.0;

    match strategy {
        TpStrategy::Replicate => TpTensorMeta {
            load_offsets: data_offsets,
            sliced_shape: shape.to_vec(),
            narrow: None,
        },
        TpStrategy::Colwise => {
            // Colwise: slice dim -2 for 2D+, dim -1 for 1D
            let slice_dim = if ndim == 1 { 0 } else { ndim - 2 };
            let dim_size = shape[slice_dim];
            let chunk = dim_size.div_ceil(world_size);
            let start_idx = rank * chunk;
            let end_idx = std::cmp::min(start_idx + chunk, dim_size);
            let slice_size = end_idx - start_idx;

            // Compute byte offset: elements after slice_dim are contiguous
            let inner_elems: usize = shape[slice_dim + 1..].iter().product::<usize>().max(1);
            let row_bytes = (inner_elems * elem_bits).div_ceil(8);
            let byte_start = data_start + start_idx * row_bytes;
            let byte_end = data_start + end_idx * row_bytes;

            let mut sliced_shape = shape.to_vec();
            sliced_shape[slice_dim] = slice_size;

            TpTensorMeta {
                load_offsets: (byte_start, byte_end),
                sliced_shape,
                narrow: None,
            }
        }
        TpStrategy::Rowwise => {
            if ndim <= 1 {
                // 1D bias: replicated for rowwise
                return TpTensorMeta {
                    load_offsets: data_offsets,
                    sliced_shape: shape.to_vec(),
                    narrow: None,
                };
            }
            // 2D+: slice dim -1, but non-contiguous → load full tensor, narrow later
            let slice_dim = ndim - 1;
            let dim_size = shape[slice_dim];
            let chunk = dim_size.div_ceil(world_size);
            let start_idx = rank * chunk;
            let end_idx = std::cmp::min(start_idx + chunk, dim_size);
            let slice_size = end_idx - start_idx;

            let mut sliced_shape = shape.to_vec();
            sliced_shape[slice_dim] = slice_size;

            TpTensorMeta {
                load_offsets: data_offsets, // full range (non-contiguous)
                sliced_shape,
                narrow: Some((slice_dim, start_idx, slice_size)),
            }
        }
    }
}

struct Open {
    /// Unified loader handling all I/O strategies (mmap, io_uring scatter, cuFile).
    model_loader: ModelLoader,
    framework: Framework,
    device_spec: DeviceSpec,
    /// Per-tensor TP metadata. Empty if no tp_plan was provided.
    tp_tensor_meta: HashMap<String, TpTensorMeta>,
    /// Per-shard metadata: needed for get_slice() (TensorInfo) and metadata() (freeform dict).
    shard_metadata: Vec<(usize, Metadata)>,
}

impl Open {
    fn new(
        py: Python<'_>,
        filename: PathBuf,
        framework: Framework,
        device_spec: Option<DeviceSpec>,
        backend: Option<Backend>,
        tp_plan: Option<TpPlan>,
    ) -> PyResult<Self> {
        if !filename.exists() {
            return Err(PyFileNotFoundError::new_err(format!(
                "No such file or directory: {}",
                filename.display()
            )));
        }

        // Detect sharded models: index.json files or directories
        let fname_str = filename.to_string_lossy();
        if fname_str.ends_with(".index.json") {
            return Self::new_sharded(py, filename, framework, device_spec, backend, tp_plan);
        }
        if filename.is_dir() {
            let index_path = filename.join("model.safetensors.index.json");
            if index_path.exists() {
                return Self::new_sharded(py, index_path, framework, device_spec, backend, tp_plan);
            }
            let single_path = filename.join("model.safetensors");
            if single_path.exists() {
                return Self::new(py, single_path, framework, device_spec, backend, tp_plan);
            }
            return Err(PyFileNotFoundError::new_err(format!(
                "No safetensors files found in directory: {}",
                filename.display()
            )));
        }

        let device_spec = device_spec.unwrap_or_default();
        let device = device_spec.default_device();

        validate_device_framework(&framework, &device_spec, &device)?;

        let (n, metadata) = SafeTensors::read_metadata_from_file(&filename).map_err(|e| {
            SafetensorError::new_err(format!("Error while deserializing header: {e}"))
        })?;

        let offset = n + 8;
        init_framework_modules(py, &framework, &device_spec, &device)?;

        // Precompute TP tensor metadata (cheap arithmetic, no I/O)
        let tp_tensor_meta = if let Some(ref tp) = tp_plan {
            let mut meta = HashMap::new();
            for (name, info) in metadata.tensors() {
                if let Some(strategy) = tp.resolve(&name) {
                    meta.insert(
                        name,
                        compute_tp_tensor_meta(
                            &info.shape,
                            info.dtype,
                            info.data_offsets,
                            strategy,
                            tp.rank,
                            tp.world_size,
                        ),
                    );
                }
            }
            meta
        } else {
            HashMap::new()
        };

        // Build byte range overrides from TP metadata
        let byte_range_overrides: HashMap<String, (usize, usize)> = tp_tensor_meta
            .iter()
            .map(|(name, meta)| (name.clone(), meta.load_offsets))
            .collect();

        let device_map = device_spec_to_device_map(&device_spec, &framework);
        let loader_backend = backend.unwrap_or_default().to_loader();

        let shard_metadata = vec![(offset, metadata.clone())];

        // Release the GIL during expensive I/O (loader creation + bulk preload).
        let model_loader = py
            .allow_threads(move || {
                LoaderBuilder::single(filename, offset, metadata)
                    .device_map(device_map)
                    .backend(loader_backend)
                    .byte_range_overrides(byte_range_overrides)
                    .build()
                    .map_err(|e| format!("Failed to create loader: {e}"))
            })
            .map_err(SafetensorError::new_err)?;

        Ok(Self {
            model_loader,
            framework,
            device_spec,
            tp_tensor_meta,
            shard_metadata,
        })
    }

    /// Open a sharded model from an index.json file.
    fn new_sharded(
        py: Python<'_>,
        index_path: PathBuf,
        framework: Framework,
        device_spec: Option<DeviceSpec>,
        backend: Option<Backend>,
        tp_plan: Option<TpPlan>,
    ) -> PyResult<Self> {
        let model_dir = index_path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .to_path_buf();

        let index = SafetensorsIndex::load(&index_path)
            .map_err(|e| SafetensorError::new_err(format!("Failed to load index file: {e}")))?;

        // Get unique shard filenames (sorted for deterministic ordering)
        let shard_names: Vec<String> = {
            let mut names: Vec<_> = index.shards().into_iter().map(|s| s.to_string()).collect();
            names.sort();
            names
        };

        let device_spec = device_spec.unwrap_or_default();
        let device = device_spec.default_device();

        validate_device_framework(&framework, &device_spec, &device)?;

        // Read metadata from all shard files (sequential — just header parsing)
        let mut shard_metadata: Vec<(usize, Metadata)> = Vec::with_capacity(shard_names.len());
        for shard_name in &shard_names {
            let shard_path = model_dir.join(shard_name);
            if !shard_path.exists() {
                return Err(PyFileNotFoundError::new_err(format!(
                    "Shard file not found: {}",
                    shard_path.display()
                )));
            }
            let (n, metadata) = SafeTensors::read_metadata_from_file(&shard_path).map_err(|e| {
                SafetensorError::new_err(format!("Error reading shard {shard_name}: {e}"))
            })?;
            shard_metadata.push((n + 8, metadata));
        }

        init_framework_modules(py, &framework, &device_spec, &device)?;

        let shard_paths: Vec<PathBuf> = shard_names
            .iter()
            .map(|name| model_dir.join(name))
            .collect();

        // Precompute TP tensor metadata across all shards (cheap arithmetic, no I/O)
        let tp_tensor_meta: HashMap<String, TpTensorMeta> = if let Some(ref tp) = tp_plan {
            let mut meta = HashMap::new();
            for (_offset, shard_md) in &shard_metadata {
                for (name, info) in shard_md.tensors() {
                    if let Some(strategy) = tp.resolve(&name) {
                        meta.insert(
                            name,
                            compute_tp_tensor_meta(
                                &info.shape,
                                info.dtype,
                                info.data_offsets,
                                strategy,
                                tp.rank,
                                tp.world_size,
                            ),
                        );
                    }
                }
            }
            meta
        } else {
            HashMap::new()
        };

        // Build byte range overrides from TP metadata
        let byte_range_overrides: HashMap<String, (usize, usize)> = tp_tensor_meta
            .iter()
            .map(|(name, meta)| (name.clone(), meta.load_offsets))
            .collect();

        let device_map = device_spec_to_device_map(&device_spec, &framework);
        let loader_backend = backend.unwrap_or_default().to_loader();

        // Clone shard_metadata for the Open struct (original consumed by LoaderBuilder)
        let open_shard_metadata = shard_metadata.clone();

        // Release GIL and load via unified Loader (handles scatter_load internally)
        let model_loader = py
            .allow_threads(move || {
                LoaderBuilder::sharded(shard_paths, shard_metadata)
                    .device_map(device_map)
                    .backend(loader_backend)
                    .byte_range_overrides(byte_range_overrides)
                    .build()
                    .map_err(|e| format!("Failed to load shards: {e}"))
            })
            .map_err(SafetensorError::new_err)?;

        Ok(Self {
            model_loader,
            framework,
            device_spec,
            tp_tensor_meta,
            shard_metadata: open_shard_metadata,
        })
    }

    /// Return the special non tensor information in the header
    ///
    /// Returns:
    ///     (`Dict[str, str]`):
    ///         The freeform metadata.
    pub fn metadata(&self) -> Option<HashMap<String, String>> {
        let mut merged: Option<HashMap<String, String>> = None;
        for (_, metadata) in &self.shard_metadata {
            if let Some(m) = metadata.metadata() {
                merged.get_or_insert_with(HashMap::new).extend(m.clone());
            }
        }
        merged
    }

    /// Returns the names of the tensors in the file.
    ///
    /// Returns:
    ///     (`List[str]`):
    ///         The name of the tensors contained in that file
    pub fn keys(&self) -> PyResult<Vec<String>> {
        let mut keys: Vec<String> = self
            .model_loader
            .tensor_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        keys.sort();
        Ok(keys)
    }

    /// Returns the names of the tensors in the file, ordered by offset.
    ///
    /// Returns:
    ///     (`List[str]`):
    ///         The name of the tensors contained in that file
    pub fn offset_keys(&self) -> PyResult<Vec<String>> {
        Ok(self
            .model_loader
            .tensor_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect())
    }

    /// Returns a full tensor
    ///
    /// Args:
    ///     name (`str`):
    ///         The name of the tensor you want
    ///
    /// Returns:
    ///     (`Tensor`):
    ///         The tensor in the framework you opened the file for.
    ///
    /// Example:
    /// ```python
    /// from safetensors import safe_open
    ///
    /// with safe_open("model.safetensors", framework="pt", device=0) as f:
    ///     tensor = f.get_tensor("embedding")
    ///
    /// ```
    pub fn get_tensor(&self, py: Python<'_>, name: &str) -> PyResult<PyObject> {
        let meta = self
            .model_loader
            .tensor_meta(name)
            .ok_or_else(|| {
                SafetensorError::new_err(format!("File does not contain tensor {name}"))
            })?;
        let dtype = meta.dtype;
        let original_shape = meta.shape.clone();
        let original_nbytes = meta.data_offsets.1 - meta.data_offsets.0;

        // Resolve TP metadata for shape/narrow
        let tp_meta = self.tp_tensor_meta.get(name);
        let narrow_spec = tp_meta.and_then(|tp| tp.narrow);
        let shape = if let Some(tp) = tp_meta {
            adjust_f4_shape(&self.framework, dtype, tp.sliced_shape.clone())?
        } else {
            adjust_f4_shape(&self.framework, dtype, original_shape.clone())?
        };

        // For rowwise TP, create tensor with FULL shape first, then narrow.
        let (create_shape, create_nbytes) = if narrow_spec.is_some() {
            (original_shape, original_nbytes)
        } else {
            let load_nbytes = meta.load_offsets.1 - meta.load_offsets.0;
            (shape.clone(), load_nbytes)
        };

        // Get buffer via unified cascade (per_tensor → per_device → preloaded → fetch)
        let buffer_ref = self.model_loader.get_buffer(name).map_err(|e| {
            SafetensorError::new_err(format!("Buffer fetch error: {e}"))
        })?;

        let target_device = self.device_spec.resolve(name);

        // CUDA path: create tensor via DLPack zero-copy
        if matches!(
            self.framework,
            Framework::Pytorch | Framework::Flax | Framework::Paddle
        ) {
            if let Device::Cuda(target_idx) = target_device {
                let maybe_tensor = match buffer_ref {
                    BufferRef::Whole(buf)
                        if matches!(buf.device(), LoaderDevice::Cuda(_)) =>
                    {
                        Some(buffer_to_cuda_tensor(
                            py,
                            GpuBufferRef::Shared(buf),
                            0,
                            create_nbytes,
                            dtype,
                            &create_shape,
                            &self.framework,
                            target_idx,
                        )?)
                    }
                    BufferRef::Slice {
                        buffer,
                        offset,
                        ..
                    } if matches!(buffer.device(), LoaderDevice::Cuda(_)) => {
                        Some(buffer_to_cuda_tensor(
                            py,
                            GpuBufferRef::Shared(buffer),
                            offset,
                            create_nbytes,
                            dtype,
                            &create_shape,
                            &self.framework,
                            target_idx,
                        )?)
                    }
                    BufferRef::Fetched(buf)
                        if matches!(buf.device(), LoaderDevice::Cuda(_)) =>
                    {
                        Some(buffer_to_cuda_tensor(
                            py,
                            GpuBufferRef::Owned(buf),
                            0,
                            create_nbytes,
                            dtype,
                            &create_shape,
                            &self.framework,
                            target_idx,
                        )?)
                    }
                    _ => None,
                };

                if let Some(mut t) = maybe_tensor {
                    if let Some((dim, narrow_start, narrow_len)) = narrow_spec {
                        let torch = TORCH_MODULE.get().ok_or_else(|| {
                            SafetensorError::new_err("torch module not initialized")
                        })?;
                        let torch_ref = torch.bind(py);
                        t = torch_ref
                            .getattr(intern!(py, "narrow"))?
                            .call1((&t, dim, narrow_start, narrow_len))?
                            .call_method0(intern!(py, "contiguous"))?
                            .into();
                    }
                    return Ok(t);
                }
            }
        }

        // CPU path: fetch via mmap (zero-copy) or fetch_to_vec
        let shard_loader = self.model_loader.shard_loader(name).ok_or_else(|| {
            SafetensorError::new_err(format!("No loader for tensor {name}"))
        })?;
        let shard_offset = self.model_loader.shard_offset(name).unwrap_or(0);
        let load_start = meta.load_offsets.0 + shard_offset;
        let load_end = if narrow_spec.is_some() {
            meta.data_offsets.1 + shard_offset
        } else {
            meta.load_offsets.1 + shard_offset
        };

        let mut tensor = fetch_cpu_tensor(
            shard_loader,
            py,
            load_start,
            load_end,
            dtype,
            &create_shape,
            &self.framework,
            &target_device,
        )?;

        if let Some((dim, narrow_start, narrow_len)) = narrow_spec {
            let torch = TORCH_MODULE
                .get()
                .ok_or_else(|| SafetensorError::new_err("torch module not initialized"))?;
            let torch_ref = torch.bind(py);
            tensor = torch_ref
                .getattr(intern!(py, "narrow"))?
                .call1((&tensor, dim, narrow_start, narrow_len))?
                .call_method0(intern!(py, "contiguous"))?
                .into();
        }

        Ok(tensor)
    }

    /// Returns a full slice view object
    ///
    /// Args:
    ///     name (`str`):
    ///         The name of the tensor you want
    ///
    /// Returns:
    ///     (`PySafeSlice`):
    ///         A dummy object you can slice into to get a real tensor
    /// Example:
    /// ```python
    /// from safetensors import safe_open
    ///
    /// with safe_open("model.safetensors", framework="pt", device=0) as f:
    ///     tensor_part = f.get_slice("embedding")[:, ::8]
    ///
    /// ```
    pub fn get_slice(&self, name: &str) -> PyResult<PySafeSlice> {
        let shard_idx = self.model_loader.shard_index(name).ok_or_else(|| {
            SafetensorError::new_err(format!("File does not contain tensor {name}"))
        })?;
        let (offset, metadata) = &self.shard_metadata[shard_idx];
        let info = metadata.info(name).ok_or_else(|| {
            SafetensorError::new_err(format!("Shard does not contain tensor {name}"))
        })?;
        let loader = self.model_loader.shard_loader(name).unwrap().clone();

        let device = self.device_spec.resolve(name);
        Ok(PySafeSlice {
            info: info.clone(),
            framework: self.framework.clone(),
            offset: *offset,
            device,
            loader,
        })
    }

    /// Batch-load multiple tensors.
    ///
    /// Routes through get_tensor() which uses ModelLoader's buffer cascade
    /// (per_tensor → per_device → preloaded → fetch) for efficient loading.
    pub fn get_tensors(&self, py: Python<'_>, names: Vec<String>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for name in &names {
            let tensor = self.get_tensor(py, name)?;
            dict.set_item(name, tensor)?;
        }
        Ok(dict.into())
    }
}

/// Opens a safetensors lazily and returns tensors as asked
///
/// Args:
///     filename (`str`, or `os.PathLike`):
///         The filename to open
///
///     framework (`str`):
///         The framework you want you tensors in. Supported values:
///         `pt`, `tf`, `flax`, `numpy`.
///
///     device (`str`, defaults to `"cpu"`):
///         The device on which you want the tensors.
///
///     backend (`str`, *optional*):
///         The loader backend to use. Supported values:
///         `auto` (default), `mmap`, `io_uring` (Linux only).
#[pyclass]
#[allow(non_camel_case_types)]
struct safe_open {
    inner: Option<Open>,
}

impl safe_open {
    fn inner(&self) -> PyResult<&Open> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| SafetensorError::new_err("File is closed".to_string()))?;
        Ok(inner)
    }

    fn inner_mut(&mut self) -> PyResult<&mut Open> {
        let inner = self
            .inner
            .as_mut()
            .ok_or_else(|| SafetensorError::new_err("File is closed".to_string()))?;
        Ok(inner)
    }
}

#[pymethods]
impl safe_open {
    #[new]
    #[pyo3(signature = (filename, framework, device=None, backend=None, tp_plan=None, tp_rank=None, tp_world_size=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        filename: PathBuf,
        framework: Framework,
        device: Option<DeviceSpec>,
        backend: Option<Backend>,
        tp_plan: Option<HashMap<String, String>>,
        tp_rank: Option<usize>,
        tp_world_size: Option<usize>,
    ) -> PyResult<Self> {
        // Validate and compile TP plan
        let compiled_tp = match tp_plan {
            Some(raw_plan) => {
                let rank = tp_rank.ok_or_else(|| {
                    SafetensorError::new_err("tp_rank is required when tp_plan is provided")
                })?;
                let world_size = tp_world_size.ok_or_else(|| {
                    SafetensorError::new_err("tp_world_size is required when tp_plan is provided")
                })?;
                if world_size < 2 {
                    return Err(SafetensorError::new_err(format!(
                        "tp_world_size must be >= 2, got {world_size}"
                    )));
                }
                if rank >= world_size {
                    return Err(SafetensorError::new_err(format!(
                        "tp_rank ({rank}) must be < tp_world_size ({world_size})"
                    )));
                }
                if framework != Framework::Pytorch {
                    return Err(SafetensorError::new_err(
                        "tp_plan is only supported for PyTorch framework",
                    ));
                }
                Some(
                    TpPlan::compile(&raw_plan, rank, world_size)
                        .map_err(SafetensorError::new_err)?,
                )
            }
            None => None,
        };
        let inner = Some(Open::new(
            py,
            filename,
            framework,
            device,
            backend,
            compiled_tp,
        )?);
        Ok(Self { inner })
    }

    /// Return the special non tensor information in the header
    ///
    /// Returns:
    ///     (`Dict[str, str]`):
    ///         The freeform metadata.
    pub fn metadata(&self) -> PyResult<Option<HashMap<String, String>>> {
        Ok(self.inner()?.metadata())
    }

    /// Returns the names of the tensors in the file.
    ///
    /// Returns:
    ///     (`List[str]`):
    ///         The name of the tensors contained in that file
    pub fn keys(&self) -> PyResult<Vec<String>> {
        self.inner()?.keys()
    }

    /// Returns the names of the tensors in the file, ordered by offset.
    ///
    /// Returns:
    ///     (`List[str]`):
    ///         The name of the tensors contained in that file
    pub fn offset_keys(&self) -> PyResult<Vec<String>> {
        self.inner()?.offset_keys()
    }

    /// Returns a full tensor
    ///
    /// Args:
    ///     name (`str`):
    ///         The name of the tensor you want
    ///
    /// Returns:
    ///     (`Tensor`):
    ///         The tensor in the framework you opened the file for.
    ///
    /// Example:
    /// ```python
    /// from safetensors import safe_open
    ///
    /// with safe_open("model.safetensors", framework="pt", device=0) as f:
    ///     tensor = f.get_tensor("embedding")
    ///
    /// ```
    pub fn get_tensor(&self, py: Python<'_>, name: &str) -> PyResult<PyObject> {
        self.inner()?.get_tensor(py, name)
    }

    /// Returns a full slice view object
    ///
    /// Args:
    ///     name (`str`):
    ///         The name of the tensor you want
    ///
    /// Returns:
    ///     (`PySafeSlice`):
    ///         A dummy object you can slice into to get a real tensor
    /// Example:
    /// ```python
    /// from safetensors import safe_open
    ///
    /// with safe_open("model.safetensors", framework="pt", device=0) as f:
    ///     tensor_part = f.get_slice("embedding")[:, ::8]
    ///
    /// ```
    pub fn get_slice(&self, name: &str) -> PyResult<PySafeSlice> {
        self.inner()?.get_slice(name)
    }

    /// Batch-load multiple tensors using vectorized I/O.
    ///
    /// Significantly faster than calling `get_tensor` in a loop because
    /// all I/O is batched per device into single fetchv calls.
    ///
    /// Args:
    ///     names (`List[str]`):
    ///         The names of the tensors to load
    ///
    /// Returns:
    ///     (`Dict[str, Tensor]`):
    ///         Dictionary mapping tensor names to tensors.
    pub fn get_tensors(&self, py: Python<'_>, names: Vec<String>) -> PyResult<PyObject> {
        self.inner()?.get_tensors(py, names)
    }

    /// Returns a streaming iterator over tensors using io_uring scatter.
    ///
    /// Yields `(name, tensor)` pairs as each tensor finishes its GPU transfer,
    /// allowing the consumer to process tensors while the rest are still loading.
    /// First tensor arrives in ~10ms instead of waiting for the full scatter.
    ///
    /// Only available for multi-device CUDA sharded models with io_uring.
    /// Falls back to eager get_tensor() iteration for other configurations.
    ///
    /// Args:
    ///     prefetch_count (`int`, defaults to `16`):
    ///         Bounded channel capacity for back-pressure.
    ///
    /// Returns:
    ///     Iterator yielding `(name, tensor)` pairs.
    ///
    /// Example:
    /// ```python
    /// from safetensors import safe_open
    ///
    /// with safe_open("model.safetensors.index.json", framework="pt", device={...}) as f:
    ///     for name, tensor in f.iter_tensors(prefetch_count=16):
    ///         model.load_weight(name, tensor)
    /// ```
    #[pyo3(signature = (prefetch_count=16))]
    pub fn iter_tensors(&mut self, prefetch_count: usize) -> PyResult<TensorIterator> {
        let inner = self.inner_mut()?;
        TensorIterator::new(inner, prefetch_count)
    }

    /// Start the context manager
    pub fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Exits the context manager
    pub fn __exit__(&mut self, _exc_type: PyObject, _exc_value: PyObject, _traceback: PyObject) {
        self.inner = None;
    }
}

/// Metadata for a tensor in the streaming iterator.
struct TensorIterMeta {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    /// For rowwise TP: (dim, start, length) to narrow after tensor creation.
    narrow: Option<(usize, usize, usize)>,
    /// Full tensor shape (before narrowing), for rowwise TP.
    full_shape: Option<Vec<usize>>,
    /// Full tensor nbytes, for rowwise TP.
    full_nbytes: Option<usize>,
}

/// Streaming tensor iterator backed by io_uring scatter.
///
/// Yields `(name, tensor)` pairs as each tensor finishes its GPU transfer.
/// For non-CUDA or single-device configs, falls back to eager get_tensor() iteration.
#[pyclass(unsendable)]
struct TensorIterator {
    /// Channel receiver for streaming TensorReady results.
    receiver: Option<std::sync::mpsc::Receiver<std::result::Result<TensorReady, String>>>,
    /// Framework for tensor creation.
    framework: Framework,
    /// Maps (shard_idx, data_offset) → tensor metadata for the streaming path.
    tensor_meta: HashMap<(usize, usize), TensorIterMeta>,
    /// Background scatter handle — joined on drop to collect ShardLoadResults.
    #[allow(dead_code)]
    handle: Option<std::thread::JoinHandle<safetensors::loader::Result<Vec<safetensors::loader::ShardLoadResult>>>>,
    /// Fallback: ordered list of tensor names for eager iteration.
    fallback_keys: Option<Vec<String>>,
    fallback_index: usize,
    /// Reference to Open for fallback get_tensor() calls.
    fallback_open: Option<*const Open>,
    /// Total number of tensors (for __len__).
    total: usize,
    emitted: usize,
}

// SAFETY: TensorIterator's fallback_open pointer is derived from a Python-owned
// safe_open object that outlives the iterator (the iterator borrows self).
// The receiver and handle are both Send.
unsafe impl Send for TensorIterator {}

impl TensorIterator {
    fn new(open: &mut Open, prefetch_count: usize) -> PyResult<Self> {
        let _ = prefetch_count; // used below in cfg block

        let total = open.model_loader.len();

        // Try streaming path: multi-device CUDA with io_uring
        #[cfg(all(target_os = "linux", feature = "io_uring"))]
        if let Some((rx, handle)) = open.model_loader.iter_start(prefetch_count) {
            // Build tensor_meta from shard metadata + tp_tensor_meta
            let mut tensor_meta = HashMap::new();
            for (shard_idx, (_offset, metadata)) in open.shard_metadata.iter().enumerate() {
                for (name, info) in metadata.tensors() {
                    let tp = open.tp_tensor_meta.get(&name);
                    let (data_offset, dtype, shape, narrow, full_shape, full_nbytes) =
                        if let Some(tp_meta) = tp {
                            let narrow = tp_meta.narrow;
                            let (fs, fnb) = if narrow.is_some() {
                                (Some(info.shape.to_vec()), Some(info.data_offsets.1 - info.data_offsets.0))
                            } else {
                                (None, None)
                            };
                            (tp_meta.load_offsets.0, info.dtype, tp_meta.sliced_shape.clone(), narrow, fs, fnb)
                        } else {
                            (info.data_offsets.0, info.dtype, info.shape.to_vec(), None, None, None)
                        };
                    tensor_meta.insert(
                        (shard_idx, data_offset),
                        TensorIterMeta { name, dtype, shape, narrow, full_shape, full_nbytes },
                    );
                }
            }

            return Ok(Self {
                receiver: Some(rx),
                framework: open.framework.clone(),
                tensor_meta,
                handle: Some(handle),
                fallback_keys: None,
                fallback_index: 0,
                fallback_open: None,
                total,
                emitted: 0,
            });
        }

        // Fallback: eager iteration via get_tensor()
        let keys: Vec<String> = open
            .model_loader
            .tensor_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        Ok(Self {
            receiver: None,
            framework: open.framework.clone(),
            tensor_meta: HashMap::new(),
            handle: None,
            fallback_keys: Some(keys),
            fallback_index: 0,
            fallback_open: Some(open as *const Open),
            total,
            emitted: 0,
        })
    }
}

#[pymethods]
impl TensorIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<(String, PyObject)>> {
        // Streaming path
        if let Some(ref receiver) = self.receiver {
            match receiver.recv() {
                Ok(Ok(ready)) => {
                    let key = (ready.shard_idx, ready.data_offset);
                    let meta = self.tensor_meta.get(&key).ok_or_else(|| {
                        SafetensorError::new_err(format!(
                            "Unknown tensor at shard {} offset {}",
                            ready.shard_idx, ready.data_offset
                        ))
                    })?;

                    // Determine shape/nbytes for tensor creation
                    let (create_shape, create_nbytes) = if meta.narrow.is_some() {
                        // Rowwise: create full tensor, narrow later
                        (
                            meta.full_shape.as_ref().unwrap().clone(),
                            meta.full_nbytes.unwrap(),
                        )
                    } else {
                        let nbytes = ready.buffer.len();
                        (meta.shape.clone(), nbytes)
                    };

                    let shape = adjust_f4_shape(&self.framework, meta.dtype, create_shape)?;

                    let mut tensor = cuda_tensor_from_preloaded(
                        py,
                        &ready.buffer,
                        0,
                        create_nbytes,
                        meta.dtype,
                        &shape,
                        &self.framework,
                        ready.device_idx,
                    )?;

                    // Apply rowwise narrow if needed
                    if let Some((dim, narrow_start, narrow_len)) = meta.narrow {
                        let torch = TORCH_MODULE.get().ok_or_else(|| {
                            SafetensorError::new_err("torch module not initialized")
                        })?;
                        let torch_ref = torch.bind(py);
                        tensor = torch_ref
                            .getattr(intern!(py, "narrow"))?
                            .call1((&tensor, dim, narrow_start, narrow_len))?
                            .call_method0(intern!(py, "contiguous"))?
                            .into();
                    }

                    self.emitted += 1;
                    return Ok(Some((meta.name.clone(), tensor)));
                }
                Ok(Err(e)) => {
                    return Err(SafetensorError::new_err(format!("Scatter error: {e}")));
                }
                Err(_) => {
                    // Channel closed — iteration complete
                    self.receiver = None;
                    return Ok(None);
                }
            }
        }

        // Fallback path: eager get_tensor()
        if let (Some(ref keys), Some(open_ptr)) = (&self.fallback_keys, self.fallback_open) {
            if self.fallback_index >= keys.len() {
                return Ok(None);
            }
            let name = &keys[self.fallback_index];
            self.fallback_index += 1;
            self.emitted += 1;

            // SAFETY: open_ptr points to the Open inside the safe_open that created
            // this iterator. The iterator cannot outlive the safe_open (Python holds
            // both, and the iterator is created from &mut self which borrows the safe_open).
            let open = unsafe { &*open_ptr };
            let tensor = open.get_tensor(py, name)?;
            return Ok(Some((name.clone(), tensor)));
        }

        Ok(None)
    }

    fn __len__(&self) -> usize {
        self.total.saturating_sub(self.emitted)
    }
}

#[pyclass]
struct PySafeSlice {
    info: TensorInfo,
    framework: Framework,
    offset: usize,
    device: Device,
    loader: Arc<FileLoader>,
}

#[derive(FromPyObject)]
enum SliceIndex<'a> {
    Slice(PyBound<'a, PySlice>),
    Index(i32),
}

#[derive(FromPyObject)]
enum Slice<'a> {
    Slice(SliceIndex<'a>),
    Slices(Vec<SliceIndex<'a>>),
}

use std::fmt;
struct Disp(Vec<TensorIndexer>);

/// Should be more readable that the standard
/// `Debug`
impl fmt::Display for Disp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, item) in self.0.iter().enumerate() {
            write!(f, "{prefix}{item}", prefix = if i == 0 { "" } else { ", " })?;
        }
        write!(f, "]")
    }
}

#[pymethods]
impl PySafeSlice {
    /// Returns the shape of the full underlying tensor
    ///
    /// Returns:
    ///     (`List[int]`):
    ///         The shape of the full tensor
    ///
    /// Example:
    /// ```python
    /// from safetensors import safe_open
    ///
    /// with safe_open("model.safetensors", framework="pt", device=0) as f:
    ///     tslice = f.get_slice("embedding")
    ///     shape = tslice.get_shape()
    ///     dim = shape // 8
    ///     tensor = tslice[:, :dim]
    /// ```
    pub fn get_shape(&self, py: Python) -> PyResult<PyObject> {
        let shape = self.info.shape.clone();
        let shape: PyObject = shape.into_pyobject(py)?.into();
        Ok(shape)
    }

    /// Returns the dtype of the full underlying tensor
    ///
    /// Returns:
    ///     (`str`):
    ///         The dtype of the full tensor
    ///
    /// Example:
    /// ```python
    /// from safetensors import safe_open
    ///
    /// with safe_open("model.safetensors", framework="pt", device=0) as f:
    ///     tslice = f.get_slice("embedding")
    ///     dtype = tslice.get_dtype() # "F32"
    /// ```
    pub fn get_dtype(&self, py: Python) -> PyResult<PyObject> {
        Ok(self.info.dtype.to_string().into_pyobject(py)?.into())
    }

    pub fn __getitem__(&self, slices: &PyBound<'_, PyAny>) -> PyResult<PyObject> {
        // Handle Ellipsis (...) - means "select everything", return full tensor.
        // This is used by transformers' weight loader to materialize tensors.
        let is_ellipsis = slices.is(slices.py().Ellipsis());

        // For CUDA-capable frameworks with CUDA loader, load full tensor to GPU then slice on GPU
        // (CPU-side slicing requires CPU buffer which CUDA loader can't provide)
        // NOTE: We check the loader's device, not self.device, because the loader
        // may be configured for CPU even when target device is CUDA
        if matches!(
            self.framework,
            Framework::Pytorch | Framework::Flax | Framework::Paddle
        ) && matches!(self.loader.device(), LoaderDevice::Cuda(_))
        {
            let start = self.info.data_offsets.0 + self.offset;
            let end = self.info.data_offsets.1 + self.offset;

            let shape =
                adjust_f4_shape(&self.framework, self.info.dtype, self.info.shape.to_vec())?;

            // Load full tensor to GPU
            let device_index = match self.device {
                Device::Cuda(idx) => idx,
                _ => 0,
            };
            // Set CUDA device for multi-device support
            self.loader
                .set_cuda_device(device_index)
                .map_err(|e| SafetensorError::new_err(format!("cudaSetDevice failed: {e}")))?;
            let full_tensor = Python::with_gil(|py| {
                fetch_cuda_tensor(
                    &self.loader,
                    py,
                    start,
                    end,
                    self.info.dtype,
                    &shape,
                    &self.framework,
                    device_index,
                )
            })?;

            // Apply slice on GPU tensor
            return Python::with_gil(|py| {
                let tensor = full_tensor.bind(py);
                let sliced = tensor
                    .getattr(intern!(py, "__getitem__"))?
                    .call1((slices,))?;
                Ok(sliced.into())
            });
        }

        // CPU path: fetch data and apply slicing
        let start = self.info.data_offsets.0 + self.offset;
        let end = self.info.data_offsets.1 + self.offset;
        let buffer = self
            .loader
            .fetch(start, end)
            .map_err(|e| SafetensorError::new_err(format!("Loader fetch error: {e}")))?;

        let data = buffer
            .as_slice()
            .ok_or_else(|| SafetensorError::new_err("Buffer not on CPU"))?;

        // Ellipsis: return full tensor without slicing
        if is_ellipsis {
            return Python::with_gil(|py| {
                let array: Py<PyAny> = PyByteArray::new(py, data).into_any().into();
                create_tensor(
                    &self.framework,
                    self.info.dtype,
                    &self.info.shape,
                    array,
                    &self.device,
                )
            });
        }

        let pyslices = slices;
        let slices: Slice = pyslices.extract()?;
        let is_list = pyslices.is_instance_of::<PyList>();
        let slices: Vec<SliceIndex> = match slices {
            Slice::Slice(slice) => vec![slice],
            Slice::Slices(slices) => {
                if slices.is_empty() && is_list {
                    vec![SliceIndex::Slice(PySlice::new(pyslices.py(), 0, 0, 0))]
                } else if is_list {
                    return Err(SafetensorError::new_err(
                        "Non empty lists are not implemented",
                    ));
                } else {
                    slices
                }
            }
        };

        let shape = self.info.shape.clone();

        let tensor = TensorView::new(self.info.dtype, self.info.shape.clone(), data)
            .map_err(|e| SafetensorError::new_err(format!("Error preparing tensor view: {e}")))?;
        let slices: Vec<TensorIndexer> = slices
            .into_iter()
            .zip(shape)
            .enumerate()
            .map(slice_to_indexer)
            .collect::<Result<_, _>>()?;

        let iterator = tensor.sliced_data(&slices).map_err(|e| {
            SafetensorError::new_err(format!(
                "Error during slicing {} with shape {:?}: {e}",
                Disp(slices),
                self.info.shape,
            ))
        })?;
        let newshape = iterator.newshape();

        // For slicing, we need to copy the non-contiguous data into a new buffer
        let mut offset = 0;
        let length = iterator.remaining_byte_len();
        Python::with_gil(|py| {
            let array: PyObject = PyByteArray::new_with(py, length, |bytes: &mut [u8]| {
                for slice in iterator {
                    let len = slice.len();
                    bytes[offset..offset + slice.len()].copy_from_slice(slice);
                    offset += len;
                }
                Ok(())
            })?
            .into_any()
            .into();
            create_tensor(
                &self.framework,
                self.info.dtype,
                &newshape,
                array,
                &self.device,
            )
        })
    }
}

fn get_module<'a>(
    py: Python<'a>,
    cell: &'static OnceLock<Py<PyModule>>,
) -> PyResult<&'a PyBound<'a, PyModule>> {
    let module: &PyBound<'a, PyModule> = cell
        .get()
        .ok_or_else(|| SafetensorError::new_err("Could not find module"))?
        .bind(py);
    Ok(module)
}

/// Create a tensor from a PyByteArray using frombuffer.
///
/// This is the fallback path used for empty tensors (where DLPack can't be used
/// due to null pointers) and for the legacy slicing path.
fn create_tensor<'a>(
    framework: &'a Framework,
    dtype: Dtype,
    shape: &'a [usize],
    array: Py<PyAny>,
    device: &'a Device,
) -> PyResult<PyObject> {
    Python::with_gil(|py| -> PyResult<PyObject> {
        let (module, is_numpy): (&PyBound<'_, PyModule>, bool) = match framework {
            Framework::Pytorch => (
                TORCH_MODULE
                    .get()
                    .ok_or_else(|| {
                        SafetensorError::new_err(format!("Could not find module {framework}",))
                    })?
                    .bind(py),
                false,
            ),
            frame => {
                // Attempt to load the frameworks
                // Those are needed to prepare the ml dtypes
                // like bfloat16
                match frame {
                    Framework::Tensorflow => {
                        let _ = PyModule::import(py, intern!(py, "tensorflow"));
                    }
                    Framework::Flax => {
                        let _ = PyModule::import(py, intern!(py, "flax"));
                    }
                    Framework::Paddle => {
                        let _ = PyModule::import(py, intern!(py, "paddle"));
                    }
                    _ => {}
                };

                (get_module(py, &NUMPY_MODULE)?, true)
            }
        };
        let dtype: PyObject = get_pydtype(module, dtype, is_numpy)?;
        let count: usize = shape.iter().product();
        let shape = shape.to_vec();
        let tensor = if count == 0 {
            // Torch==1.10 does not allow frombuffer on empty buffers so we create
            // the tensor manually.
            // let zeros = module.getattr(intern!(py, "zeros"))?;
            let shape: PyObject = shape.clone().into_pyobject(py)?.into();
            let args = (shape,);
            let kwargs = [(intern!(py, "dtype"), dtype)].into_py_dict(py)?;
            module.call_method("zeros", args, Some(&kwargs))?
        } else {
            // let frombuffer = module.getattr(intern!(py, "frombuffer"))?;
            let kwargs = [
                (intern!(py, "buffer"), array),
                (intern!(py, "dtype"), dtype),
            ]
            .into_py_dict(py)?;
            let mut tensor = module.call_method("frombuffer", (), Some(&kwargs))?;
            let sys = PyModule::import(py, intern!(py, "sys"))?;
            let byteorder: String = sys.getattr(intern!(py, "byteorder"))?.extract()?;
            if byteorder == "big" {
                let inplace_kwargs =
                    [(intern!(py, "inplace"), PyBool::new(py, false))].into_py_dict(py)?;
                tensor = tensor
                    .getattr("byteswap")?
                    .call((), Some(&inplace_kwargs))?;
            }
            tensor
        };
        let mut tensor: PyBound<'_, PyAny> = tensor.call_method1("reshape", (shape,))?;
        let tensor = match framework {
            Framework::Flax => {
                let module = Python::with_gil(|py| -> PyResult<&Py<PyModule>> {
                    let module = PyModule::import(py, intern!(py, "jax"))?;
                    Ok(FLAX_MODULE.get_or_init_py_attached(py, || module.into()))
                })?
                .bind(py);
                module
                    .getattr(intern!(py, "numpy"))?
                    .getattr(intern!(py, "array"))?
                    .call1((tensor,))?
            }
            Framework::Tensorflow => {
                let module = Python::with_gil(|py| -> PyResult<&Py<PyModule>> {
                    let module = PyModule::import(py, intern!(py, "tensorflow"))?;
                    Ok(TENSORFLOW_MODULE.get_or_init_py_attached(py, || module.into()))
                })?
                .bind(py);
                module
                    .getattr(intern!(py, "convert_to_tensor"))?
                    .call1((tensor,))?
            }
            Framework::Mlx => {
                let module = Python::with_gil(|py| -> PyResult<&Py<PyModule>> {
                    let module = PyModule::import(py, intern!(py, "mlx"))?;
                    Ok(MLX_MODULE.get_or_init_py_attached(py, || module.into()))
                })?
                .bind(py);
                module
                    .getattr(intern!(py, "core"))?
                    // .getattr(intern!(py, "array"))?
                    .call_method1("array", (tensor,))?
            }
            Framework::Paddle => {
                let module = Python::with_gil(|py| -> PyResult<&Py<PyModule>> {
                    let module = PyModule::import(py, intern!(py, "paddle"))?;
                    Ok(PADDLE_MODULE.get_or_init_py_attached(py, || module.into()))
                })?
                .bind(py);
                let device: PyObject = if let Device::Cuda(index) = device {
                    format!("gpu:{index}").into_pyobject(py)?.into()
                } else {
                    device.clone().into_pyobject(py)?.into()
                };
                let kwargs = [(intern!(py, "place"), device)].into_py_dict(py)?;
                let tensor = module
                    .getattr(intern!(py, "to_tensor"))?
                    .call((tensor,), Some(&kwargs))?;
                tensor
            }
            Framework::Pytorch => {
                if device != &Device::Cpu {
                    let device: PyObject = device.clone().into_pyobject(py)?.into();
                    let kwargs = PyDict::new(py);
                    tensor = tensor.call_method("to", (device,), Some(&kwargs))?;
                }
                tensor
            }
            Framework::Numpy => tensor,
        };
        // let tensor = tensor.into_py_bound(py);
        Ok(tensor.into())
    })
}

/// Get JAX dtype from safetensors Dtype.
fn get_jax_dtype(jax: &PyBound<'_, PyModule>, dtype: Dtype) -> PyResult<PyObject> {
    let py = jax.py();
    let jnp = jax.getattr(intern!(py, "numpy"))?;
    let dtype_obj: PyObject = match dtype {
        Dtype::F64 => jnp.getattr(intern!(py, "float64"))?.into(),
        Dtype::F32 => jnp.getattr(intern!(py, "float32"))?.into(),
        Dtype::BF16 => jnp.getattr(intern!(py, "bfloat16"))?.into(),
        Dtype::F16 => jnp.getattr(intern!(py, "float16"))?.into(),
        Dtype::U64 => jnp.getattr(intern!(py, "uint64"))?.into(),
        Dtype::I64 => jnp.getattr(intern!(py, "int64"))?.into(),
        Dtype::U32 => jnp.getattr(intern!(py, "uint32"))?.into(),
        Dtype::I32 => jnp.getattr(intern!(py, "int32"))?.into(),
        Dtype::U16 => jnp.getattr(intern!(py, "uint16"))?.into(),
        Dtype::I16 => jnp.getattr(intern!(py, "int16"))?.into(),
        Dtype::U8 => jnp.getattr(intern!(py, "uint8"))?.into(),
        Dtype::I8 => jnp.getattr(intern!(py, "int8"))?.into(),
        Dtype::BOOL => jnp.getattr(intern!(py, "bool_"))?.into(),
        Dtype::C64 => jnp.getattr(intern!(py, "complex64"))?.into(),
        dtype => {
            return Err(SafetensorError::new_err(format!(
                "Dtype not supported in JAX: {dtype:?}"
            )))
        }
    };
    Ok(dtype_obj)
}

/// Get MLX dtype from safetensors Dtype.
fn get_mlx_dtype(mlx: &PyBound<'_, PyModule>, dtype: Dtype) -> PyResult<PyObject> {
    let py = mlx.py();
    let core = mlx.getattr(intern!(py, "core"))?;
    let dtype_obj: PyObject = match dtype {
        Dtype::F32 => core.getattr(intern!(py, "float32"))?.into(),
        Dtype::BF16 => core.getattr(intern!(py, "bfloat16"))?.into(),
        Dtype::F16 => core.getattr(intern!(py, "float16"))?.into(),
        Dtype::U64 => core.getattr(intern!(py, "uint64"))?.into(),
        Dtype::I64 => core.getattr(intern!(py, "int64"))?.into(),
        Dtype::U32 => core.getattr(intern!(py, "uint32"))?.into(),
        Dtype::I32 => core.getattr(intern!(py, "int32"))?.into(),
        Dtype::U16 => core.getattr(intern!(py, "uint16"))?.into(),
        Dtype::I16 => core.getattr(intern!(py, "int16"))?.into(),
        Dtype::U8 => core.getattr(intern!(py, "uint8"))?.into(),
        Dtype::I8 => core.getattr(intern!(py, "int8"))?.into(),
        Dtype::BOOL => core.getattr(intern!(py, "bool_"))?.into(),
        dtype => {
            return Err(SafetensorError::new_err(format!(
                "Dtype not supported in MLX: {dtype:?}"
            )))
        }
    };
    Ok(dtype_obj)
}

/// Get Paddle dtype from safetensors Dtype.
fn get_paddle_dtype(paddle: &PyBound<'_, PyModule>, dtype: Dtype) -> PyResult<PyObject> {
    let py = paddle.py();
    let dtype_obj: PyObject = match dtype {
        Dtype::F64 => paddle.getattr(intern!(py, "float64"))?.into(),
        Dtype::F32 => paddle.getattr(intern!(py, "float32"))?.into(),
        Dtype::BF16 => paddle.getattr(intern!(py, "bfloat16"))?.into(),
        Dtype::F16 => paddle.getattr(intern!(py, "float16"))?.into(),
        Dtype::U64 => paddle.getattr(intern!(py, "uint64"))?.into(),
        Dtype::I64 => paddle.getattr(intern!(py, "int64"))?.into(),
        Dtype::U32 => paddle.getattr(intern!(py, "uint32"))?.into(),
        Dtype::I32 => paddle.getattr(intern!(py, "int32"))?.into(),
        Dtype::U16 => paddle.getattr(intern!(py, "uint16"))?.into(),
        Dtype::I16 => paddle.getattr(intern!(py, "int16"))?.into(),
        Dtype::U8 => paddle.getattr(intern!(py, "uint8"))?.into(),
        Dtype::I8 => paddle.getattr(intern!(py, "int8"))?.into(),
        Dtype::BOOL => paddle.getattr(intern!(py, "bool"))?.into(),
        dtype => {
            return Err(SafetensorError::new_err(format!(
                "Dtype not supported in Paddle: {dtype:?}"
            )))
        }
    };
    Ok(dtype_obj)
}

fn get_pydtype(module: &PyBound<'_, PyModule>, dtype: Dtype, is_numpy: bool) -> PyResult<PyObject> {
    let py = module.py();
    let dtype: PyObject = match dtype {
        Dtype::F64 => module.getattr(intern!(py, "float64"))?.into(),
        Dtype::F32 => module.getattr(intern!(py, "float32"))?.into(),
        Dtype::BF16 => {
            if is_numpy {
                module
                    .getattr(intern!(py, "dtype"))?
                    .call1(("bfloat16",))?
                    .into()
            } else {
                module.getattr(intern!(py, "bfloat16"))?.into()
            }
        }
        Dtype::F16 => module.getattr(intern!(py, "float16"))?.into(),
        Dtype::U64 => module.getattr(intern!(py, "uint64"))?.into(),
        Dtype::I64 => module.getattr(intern!(py, "int64"))?.into(),
        Dtype::U32 => module.getattr(intern!(py, "uint32"))?.into(),
        Dtype::I32 => module.getattr(intern!(py, "int32"))?.into(),
        Dtype::U16 => module.getattr(intern!(py, "uint16"))?.into(),
        Dtype::I16 => module.getattr(intern!(py, "int16"))?.into(),
        Dtype::U8 => module.getattr(intern!(py, "uint8"))?.into(),
        Dtype::I8 => module.getattr(intern!(py, "int8"))?.into(),
        Dtype::BOOL => {
            if is_numpy {
                py.import("builtins")?.getattr(intern!(py, "bool"))?.into()
            } else {
                module.getattr(intern!(py, "bool"))?.into()
            }
        }
        Dtype::F8_E4M3 => {
            if is_numpy {
                module
                    .getattr(intern!(py, "dtype"))?
                    .call1(("float8_e4m3fn",))?
                    .into()
            } else {
                module.getattr(intern!(py, "float8_e4m3fn"))?.into()
            }
        }
        Dtype::F8_E5M2 => {
            if is_numpy {
                module
                    .getattr(intern!(py, "dtype"))?
                    .call1(("float8_e5m2",))?
                    .into()
            } else {
                module.getattr(intern!(py, "float8_e5m2"))?.into()
            }
        }
        Dtype::F8_E8M0 => {
            if is_numpy {
                module
                    .getattr(intern!(py, "dtype"))?
                    .call1(("float8_e8m0fnu",))?
                    .into()
            } else {
                module.getattr(intern!(py, "float8_e8m0fnu"))?.into()
            }
        }
        Dtype::F4 => {
            if is_numpy {
                module
                    .getattr(intern!(py, "dtype"))?
                    .call1(("float4_e2m1fn_x2",))?
                    .into()
            } else {
                module.getattr(intern!(py, "float4_e2m1fn_x2"))?.into()
            }
        }
        Dtype::C64 => module.getattr(intern!(py, "complex64"))?.into(),
        dtype => {
            return Err(SafetensorError::new_err(format!(
                "Dtype not understood: {dtype}"
            )))
        }
    };
    Ok(dtype)
}

pyo3::create_exception!(
    safetensors_rust,
    SafetensorError,
    PyException,
    "Custom Python Exception for Safetensor errors."
);

#[pyclass]
#[allow(non_camel_case_types)]
struct _safe_open_handle {
    inner: Option<Open>,
}

impl _safe_open_handle {
    fn inner(&self) -> PyResult<&Open> {
        let inner = self
            .inner
            .as_ref()
            .ok_or_else(|| SafetensorError::new_err("File is closed".to_string()))?;
        Ok(inner)
    }
}

#[pymethods]
impl _safe_open_handle {
    #[new]
    #[pyo3(signature = (f, framework, device=Some(Device::Cpu), backend=None))]
    fn new(
        py: Python<'_>,
        f: PyObject,
        framework: Framework,
        device: Option<Device>,
        backend: Option<Backend>,
    ) -> PyResult<Self> {
        let _ = f.getattr(py, "fileno")?;
        let filename: PathBuf = f.getattr(py, "name")?.extract(py)?;
        // Convert single device to DeviceSpec
        let device_spec = device.map(DeviceSpec::Single);
        let inner = Some(Open::new(
            py,
            filename,
            framework,
            device_spec,
            backend,
            None,
        )?);
        Ok(Self { inner })
    }

    /// Return the special non tensor information in the header
    ///
    /// Returns:
    ///     (`Dict[str, str]`):
    ///         The freeform metadata.
    pub fn metadata(&self) -> PyResult<Option<HashMap<String, String>>> {
        Ok(self.inner()?.metadata())
    }

    /// Returns the names of the tensors in the file.
    ///
    /// Returns:
    ///     (`List[str]`):
    ///         The name of the tensors contained in that file
    pub fn keys(&self) -> PyResult<Vec<String>> {
        self.inner()?.keys()
    }

    /// Returns the names of the tensors in the file, ordered by offset.
    ///
    /// Returns:
    ///     (`List[str]`):
    ///         The name of the tensors contained in that file
    pub fn offset_keys(&self) -> PyResult<Vec<String>> {
        self.inner()?.offset_keys()
    }

    /// Returns a full tensor
    ///
    /// Args:
    ///     name (`str`):
    ///         The name of the tensor you want
    ///
    /// Returns:
    ///     (`Tensor`):
    ///         The tensor in the framework you opened the file for.
    ///
    /// Example:
    /// ```python
    /// from safetensors import safe_open
    ///
    /// with safe_open("model.safetensors", framework="pt", device=0) as f:
    ///     tensor = f.get_tensor("embedding")
    ///
    /// ```
    pub fn get_tensor(&self, py: Python<'_>, name: &str) -> PyResult<PyObject> {
        self.inner()?.get_tensor(py, name)
    }

    /// Returns a full slice view object
    ///
    /// Args:
    ///     name (`str`):
    ///         The name of the tensor you want
    ///
    /// Returns:
    ///     (`PySafeSlice`):
    ///         A dummy object you can slice into to get a real tensor
    /// Example:
    /// ```python
    /// from safetensors import safe_open
    ///
    /// with safe_open("model.safetensors", framework="pt", device=0) as f:
    ///     tensor_part = f.get_slice("embedding")[:, ::8]
    ///
    /// ```
    pub fn get_slice(&self, name: &str) -> PyResult<PySafeSlice> {
        self.inner()?.get_slice(name)
    }

    /// Start the context manager
    pub fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Exits the context manager
    pub fn __exit__(&mut self, _exc_type: PyObject, _exc_value: PyObject, _traceback: PyObject) {
        self.inner = None;
    }
}

/// A Python module implemented in Rust.
#[pymodule(gil_used = false)]
fn _safetensors_rust(m: &PyBound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(serialize, m)?)?;
    m.add_function(wrap_pyfunction!(serialize_file, m)?)?;
    m.add_function(wrap_pyfunction!(deserialize, m)?)?;
    m.add_class::<safe_open>()?;
    m.add_class::<_safe_open_handle>()?;
    m.add_class::<CpuBuffer>()?;
    m.add_class::<GpuBufferHolder>()?;
    m.add_class::<TensorIterator>()?;
    m.add("SafetensorError", m.py().get_type::<SafetensorError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
