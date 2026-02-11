#![deny(missing_docs)]
//! Dummy doc
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
use safetensors::loader::{
    Backend as LoaderBackend, Buffer as LoaderBuffer, Device as LoaderDevice, Loader as CoreLoader,
    OwnedPrefetchIterator, PrefetchConfig, TensorLoadInfo,
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
    /// Create a new CpuBuffer from an hmll LoaderBuffer.
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

/// Lightweight holder to keep GPU buffer alive when attached to a tensor.
/// Attached via tensor._safetensors_buffer to prevent deallocation.
#[pyclass]
struct GpuBufferHolder {
    #[allow(dead_code)]
    buffer: LoaderBuffer,
}

/// Wrapper for creating DLPack tensor from GPU memory.
/// This struct owns all its data to avoid lifetime issues with PyO3.
struct DlpackGpuWrapper {
    ptr: *mut std::ffi::c_void,
    shape: Vec<i64>,
    dtype: dlpack_ffi::DataType,
    device_index: usize,
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

/// Fetch a range of bytes and create a CUDA tensor with ZERO COPY.
///
/// Loads data directly to GPU, then creates a tensor that wraps
/// the GPU memory via DLPack protocol. For exotic dtypes (F8, F4), exports
/// as uint8 via DLPack then view-casts to the target dtype.
///
/// A GpuBufferHolder is attached to the tensor to keep the GPU memory alive.
///
/// Supported frameworks: PyTorch, JAX/Flax, Paddle
fn fetch_cuda_tensor(
    loader: &CoreLoader,
    py: Python<'_>,
    start: usize,
    end: usize,
    dtype: Dtype,
    shape: &[usize],
    framework: &Framework,
) -> PyResult<PyObject> {
    let current_device = loader.device();
    let device_index = if let LoaderDevice::Cuda(idx) = current_device {
        idx
    } else {
        return Err(SafetensorError::new_err(format!(
            "Loader not configured for CUDA: {current_device}"
        )));
    };

    let buffer = loader
        .fetch(start, end)
        .map_err(|e| SafetensorError::new_err(format!("Loader fetch error: {e}")))?;

    let nbytes = buffer.len();
    let ptr = buffer.as_ptr() as *mut std::ffi::c_void;

    let (dlpack_dtype, dlpack_shape) = if let Some(dt) = dtype_to_dlpack(dtype) {
        let shape_i64: Vec<i64> = shape.iter().map(|&s| s as i64).collect();
        (dt, shape_i64)
    } else {
        // Export as flat uint8 array for exotic dtypes
        (dlpack_ffi::DataType::U8, vec![nbytes as i64])
    };

    let wrapper = DlpackGpuWrapper {
        ptr,
        shape: dlpack_shape,
        dtype: dlpack_dtype,
        device_index,
    };

    let managed_tensor = SafeManagedTensor::new(wrapper).map_err(|e| {
        SafetensorError::new_err(format!("Failed to create DLPack tensor: {:?}", e))
    })?;

    let capsule = managed_tensor_to_capsule(py, managed_tensor)?;

    // Create tensor from DLPack capsule using framework-specific from_dlpack
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
            Framework::Flax => {
                // JAX doesn't support exotic dtypes like F8/F4, fall back to error
                return Err(SafetensorError::new_err(format!(
                    "Exotic dtype {dtype:?} not supported in JAX"
                )));
            }
            Framework::Paddle => {
                // Paddle doesn't support exotic dtypes like F8/F4, fall back to error
                return Err(SafetensorError::new_err(format!(
                    "Exotic dtype {dtype:?} not supported in Paddle"
                )));
            }
            _ => tensor,
        }
    } else {
        tensor
    };

    // Create lightweight holder to keep buffer alive and attach to tensor
    let holder = Py::new(py, GpuBufferHolder { buffer })?;
    tensor.setattr(intern!(py, "_safetensors_buffer"), holder)?;

    Ok(tensor.into())
}

/// Fetch a range of bytes and create a CPU tensor using zero-copy via DLPack.
///
/// Creates tensors directly via DLPack protocol for potential zero-copy across frameworks.
/// Falls back to PyByteArray for empty tensors (null pointers not supported by DLPack).
#[allow(clippy::too_many_arguments)]
fn fetch_cpu_tensor(
    loader: &CoreLoader,
    py: Python<'_>,
    start: usize,
    end: usize,
    dtype: Dtype,
    shape: &[usize],
    framework: &Framework,
    device: &Device,
) -> PyResult<PyObject> {
    let buffer = loader
        .fetch_view(start, end)
        .map_err(|e| SafetensorError::new_err(format!("Loader fetch error: {e}")))?;

    // Empty tensors have null pointers - fall back to PyByteArray
    // (DLPack doesn't support null pointers)
    if buffer.as_ptr().is_null() {
        let vec = buffer
            .to_vec()
            .map_err(|e| SafetensorError::new_err(format!("Buffer conversion error: {e}")))?;
        let array: PyObject = PyByteArray::new(py, &vec).into_any().into();
        return create_tensor(framework, dtype, shape, array, device);
    }

    let cpu_buffer = Py::new(py, CpuBuffer::new(buffer))?;
    create_tensor_from_dlpack(framework, cpu_buffer, device, dtype, shape)
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
    /// User didn't specify accelerator, torch
    /// is responsible for choosing.
    Anonymous(usize),
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
            Device::Anonymous(index) => write!(f, "{index}"),
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
            Ok(Device::Anonymous(number))
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
            Device::Anonymous(n) => n.into_pyobject(py).map(|x| x.into_any()),
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
}

impl Backend {
    /// Convert to loader Backend.
    fn to_loader(self) -> LoaderBackend {
        match self {
            Backend::Auto => LoaderBackend::Auto,
            Backend::Mmap => LoaderBackend::Mmap,
            #[cfg(all(target_os = "linux", feature = "io_uring"))]
            Backend::IoUring => LoaderBackend::IoUring,
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
                "io_uring backend is only available on Linux with io_uring feature enabled"
            )),
            name => Err(SafetensorError::new_err(format!(
                "backend {name} is invalid (valid: auto, mmap, io_uring)"
            ))),
        }
    }
}

struct Open {
    metadata: Metadata,
    offset: usize,
    framework: Framework,
    device: Device,
    loader: Arc<CoreLoader>,
}

impl Open {
    fn new(
        filename: PathBuf,
        framework: Framework,
        device: Option<Device>,
        backend: Option<Backend>,
    ) -> PyResult<Self> {
        // Validate file exists
        if !filename.exists() {
            return Err(PyFileNotFoundError::new_err(format!(
                "No such file or directory: {}",
                filename.display()
            )));
        }

        let device = device.unwrap_or(Device::Cpu);
        if device != Device::Cpu
            && framework != Framework::Pytorch
            && framework != Framework::Paddle
        {
            return Err(SafetensorError::new_err(format!(
                "Device {device} is not supported for framework {framework}",
            )));
        }

        // Read metadata from the file header (efficient - only reads header, not full file)
        let (n, metadata) = SafeTensors::read_metadata_from_file(&filename).map_err(|e| {
            SafetensorError::new_err(format!("Error while deserializing header: {e}"))
        })?;

        let offset = n + 8;
        Python::with_gil(|py| -> PyResult<()> {
            // Initialize numpy - needed for DLPack path and fallback
            let numpy = PyModule::import(py, intern!(py, "numpy"))?;
            let np_from_dlpack = numpy.getattr(intern!(py, "from_dlpack"))?;
            NUMPY_FROM_DLPACK.get_or_init_py_attached(py, || np_from_dlpack.unbind());
            NUMPY_MODULE.get_or_init_py_attached(py, || numpy.into());

            // Initialize framework-specific modules and cache from_dlpack
            match framework {
                Framework::Pytorch => {
                    let module = PyModule::import(py, intern!(py, "torch"))?;
                    let from_dlpack = module.getattr(intern!(py, "from_dlpack"))?;
                    TORCH_FROM_DLPACK.get_or_init_py_attached(py, || from_dlpack.unbind());
                    // Set CUDA device once here BEFORE storing the module
                    if let Device::Cuda(idx) = device {
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
                    // TensorFlow uses numpy intermediate (no zero-copy DLPack support)
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
        })?;

        // Create high-performance loader (CUDA device already set above if needed)
        // CUDA-capable frameworks: hmll loads file directly to GPU memory via DMA
        // Other frameworks/devices: hmll loads to CPU via mmap, framework handles GPU transfer
        // Both paths use DLPack for zero-copy tensor creation
        let loader_device = match (&device, &framework) {
            (Device::Cuda(idx), Framework::Pytorch | Framework::Flax | Framework::Paddle) => {
                LoaderDevice::Cuda(*idx)
            }
            _ => LoaderDevice::Cpu,
        };
        let loader_backend = backend.unwrap_or_default().to_loader();
        let loader = CoreLoader::with_backend(&filename, loader_device, loader_backend)
            .map_err(|e| SafetensorError::new_err(format!("Failed to create loader: {e}")))?;

        Ok(Self {
            metadata,
            offset,
            framework,
            device,
            loader: Arc::new(loader),
        })
    }

    /// Return the special non tensor information in the header
    ///
    /// Returns:
    ///     (`Dict[str, str]`):
    ///         The freeform metadata.
    pub fn metadata(&self) -> Option<HashMap<String, String>> {
        self.metadata.metadata().clone()
    }

    /// Returns the names of the tensors in the file.
    ///
    /// Returns:
    ///     (`List[str]`):
    ///         The name of the tensors contained in that file
    pub fn keys(&self) -> PyResult<Vec<String>> {
        let mut keys: Vec<String> = self.metadata.tensors().keys().cloned().collect();
        keys.sort();
        Ok(keys)
    }

    /// Returns the names of the tensors in the file, ordered by offset.
    ///
    /// Returns:
    ///     (`List[str]`):
    ///         The name of the tensors contained in that file
    pub fn offset_keys(&self) -> PyResult<Vec<String>> {
        Ok(self.metadata.offset_keys())
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
        let info = self.metadata.info(name).ok_or_else(|| {
            SafetensorError::new_err(format!("File does not contain tensor {name}",))
        })?;

        let start = info.data_offsets.0 + self.offset;
        let end = info.data_offsets.1 + self.offset;

        // Handle F4 dtype shape adjustment for PyTorch
        let shape = if self.framework == Framework::Pytorch && info.dtype == Dtype::F4 {
            let mut shape = info.shape.to_vec();
            let n = shape.len();
            if shape[n - 1] % 2 != 0 {
                return Err(SafetensorError::new_err(format!(
                    "f4_x2 dtype requires that the last dim be divisible by 2 in torch: got {shape:?}",
                )));
            }
            shape[n - 1] /= 2;
            shape
        } else {
            info.shape.to_vec()
        };

        // For CUDA-capable frameworks with CUDA loader, use zero-copy path via DLPack
        // NOTE: We check the loader's device, not self.device, because the loader
        // may be configured for CPU even when target device is CUDA (e.g., cuda:1
        // falls back to CPU loader since hmll only supports cuda:0 direct loading)
        if matches!(
            self.framework,
            Framework::Pytorch | Framework::Flax | Framework::Paddle
        ) && matches!(self.loader.device(), LoaderDevice::Cuda(_))
        {
            return fetch_cuda_tensor(
                &self.loader,
                py,
                start,
                end,
                info.dtype,
                &shape,
                &self.framework,
            );
        }

        // CPU path: use zero-copy fetch_cpu_tensor
        fetch_cpu_tensor(
            &self.loader,
            py,
            start,
            end,
            info.dtype,
            &shape,
            &self.framework,
            &self.device,
        )
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
        if let Some(info) = self.metadata.info(name) {
            Ok(PySafeSlice {
                info: info.clone(),
                framework: self.framework.clone(),
                offset: self.offset,
                device: self.device.clone(),
                loader: self.loader.clone(),
            })
        } else {
            Err(SafetensorError::new_err(format!(
                "File does not contain tensor {name}",
            )))
        }
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
}

#[pymethods]
impl safe_open {
    #[new]
    #[pyo3(signature = (filename, framework, device=Some(Device::Cpu), backend=None))]
    fn new(
        filename: PathBuf,
        framework: Framework,
        device: Option<Device>,
        backend: Option<Backend>,
    ) -> PyResult<Self> {
        let inner = Some(Open::new(filename, framework, device, backend)?);
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

    /// Returns an iterator over tensors with async prefetching.
    ///
    /// This iterator prefetches tensors ahead of time, allowing I/O to overlap
    /// with tensor processing (e.g., quantization). For CUDA devices, this uses
    /// async GPU memory allocation and transfers.
    ///
    /// Args:
    ///     prefetch (`int`, defaults to `4`):
    ///         Number of tensors to prefetch ahead.
    ///
    /// Returns:
    ///     Iterator yielding `(name, tensor)` pairs.
    ///
    /// Example:
    /// ```python
    /// from safetensors import safe_open
    ///
    /// with safe_open("model.safetensors", framework="pt", device="cuda") as f:
    ///     for name, tensor in f.iter_tensors(prefetch=4):
    ///         # tensor is on GPU, next tensors loading in background
    ///         quantized = quantize(tensor)
    ///         model_state[name] = quantized
    /// ```
    #[pyo3(signature = (prefetch=4))]
    pub fn iter_tensors(&self, prefetch: usize) -> PyResult<PyTensorIterator> {
        let inner = self.inner()?;
        PyTensorIterator::new(inner, prefetch)
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

/// Iterator over tensors with async prefetching.
///
/// This iterator uses the Rust OwnedPrefetchIterator internally, which provides
/// async GPU memory allocation and transfers for CUDA devices.
#[pyclass(unsendable)]
struct PyTensorIterator {
    /// The underlying Rust iterator
    inner: Option<OwnedPrefetchIterator>,
    /// Framework for tensor creation
    framework: Framework,
    /// Target device
    device: Device,
    /// Tensor metadata for dtype/shape lookup
    metadata: Metadata,
}

impl PyTensorIterator {
    fn new(open: &Open, prefetch: usize) -> PyResult<Self> {
        // Build TensorLoadInfo for all tensors
        let tensor_names = open.metadata.offset_keys();
        let tensors: Vec<TensorLoadInfo> = tensor_names
            .iter()
            .map(|name| {
                let info = open.metadata.info(name).unwrap();
                TensorLoadInfo::new(
                    name.clone(),
                    info.data_offsets.0 + open.offset,
                    info.data_offsets.1 + open.offset,
                )
            })
            .collect();

        // Create the Rust prefetch iterator
        let config = PrefetchConfig::new(prefetch);
        let inner = OwnedPrefetchIterator::new(open.loader.clone(), tensors, config);

        Ok(Self {
            inner: Some(inner),
            framework: open.framework.clone(),
            device: open.device.clone(),
            metadata: open.metadata.clone(),
        })
    }

    /// Convert a buffer to a tensor using the appropriate framework.
    fn buffer_to_tensor(
        &self,
        py: Python<'_>,
        name: &str,
        buffer: LoaderBuffer,
    ) -> PyResult<PyObject> {
        let info = self.metadata.info(name).ok_or_else(|| {
            SafetensorError::new_err(format!("Tensor not found: {name}"))
        })?;

        // Handle F4 dtype shape adjustment for PyTorch
        let shape = if self.framework == Framework::Pytorch && info.dtype == Dtype::F4 {
            let mut shape = info.shape.to_vec();
            let n = shape.len();
            if shape[n - 1] % 2 != 0 {
                return Err(SafetensorError::new_err(format!(
                    "f4_x2 dtype requires that the last dim be divisible by 2 in torch: got {:?}",
                    shape
                )));
            }
            shape[n - 1] /= 2;
            shape
        } else {
            info.shape.to_vec()
        };

        // For CUDA buffers, use DLPack
        if matches!(self.device, Device::Cuda(_)) {
            return self.cuda_buffer_to_tensor(py, buffer, info.dtype, &shape);
        }

        // For CPU buffers, use zero-copy DLPack
        self.cpu_buffer_to_tensor(py, buffer, info.dtype, &shape)
    }

    /// Convert a CUDA buffer to a tensor.
    fn cuda_buffer_to_tensor(
        &self,
        py: Python<'_>,
        buffer: LoaderBuffer,
        dtype: Dtype,
        shape: &[usize],
    ) -> PyResult<PyObject> {
        let device_index = match self.device {
            Device::Cuda(idx) => idx,
            _ => return Err(SafetensorError::new_err("Not a CUDA device")),
        };

        let nbytes = buffer.len();
        let ptr = buffer.as_ptr() as *mut std::ffi::c_void;

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
        };

        let managed_tensor = SafeManagedTensor::new(wrapper).map_err(|e| {
            SafetensorError::new_err(format!("Failed to create DLPack tensor: {:?}", e))
        })?;

        let capsule = managed_tensor_to_capsule(py, managed_tensor)?;

        // Create tensor from DLPack
        let tensor: PyBound<'_, PyAny> = match self.framework {
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
                    "CUDA not supported for framework: {:?}",
                    self.framework
                )));
            }
        };

        // For exotic dtypes, view-cast and reshape
        let tensor = if dtype_to_dlpack(dtype).is_none() {
            match self.framework {
                Framework::Pytorch => {
                    let torch = get_module(py, &TORCH_MODULE)?;
                    let target_dtype = get_pydtype(torch, dtype, false)?;
                    let typed = tensor.call_method1(intern!(py, "view"), (target_dtype,))?;
                    typed.call_method1(intern!(py, "reshape"), (shape.to_vec(),))?
                }
                _ => tensor,
            }
        } else {
            tensor
        };

        // Keep buffer alive
        let holder = Py::new(py, GpuBufferHolder { buffer })?;
        tensor.setattr(intern!(py, "_safetensors_buffer"), holder)?;

        Ok(tensor.into())
    }

    /// Convert a CPU buffer to a tensor using DLPack.
    fn cpu_buffer_to_tensor(
        &self,
        py: Python<'_>,
        buffer: LoaderBuffer,
        dtype: Dtype,
        shape: &[usize],
    ) -> PyResult<PyObject> {
        let nbytes = buffer.len();

        // Handle empty tensors
        if nbytes == 0 || buffer.as_ptr().is_null() {
            let bytes = PyByteArray::new(py, &[]);
            return create_tensor(
                &self.framework,
                dtype,
                shape,
                bytes.unbind().into_any(),
                &self.device,
            );
        }

        // Wrap buffer in CpuBuffer for DLPack zero-copy
        let cpu_buffer = CpuBuffer::new(buffer);
        let py_buffer = Py::new(py, cpu_buffer)?;
        let bound_buffer = py_buffer.bind(py);

        // Get DLPack capsule
        let capsule = bound_buffer.call_method0(intern!(py, "__dlpack__"))?;

        // Create tensor from DLPack
        let tensor: PyBound<'_, PyAny> = match self.framework {
            Framework::Pytorch => {
                let from_dlpack = TORCH_FROM_DLPACK
                    .get()
                    .ok_or_else(|| SafetensorError::new_err("torch.from_dlpack not initialized"))?
                    .bind(py);
                from_dlpack.call1((capsule,))?
            }
            Framework::Numpy => {
                let from_dlpack = NUMPY_FROM_DLPACK
                    .get()
                    .ok_or_else(|| SafetensorError::new_err("numpy.from_dlpack not initialized"))?
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
            Framework::Mlx => {
                let from_dlpack = MLX_FROM_DLPACK
                    .get()
                    .ok_or_else(|| SafetensorError::new_err("mlx.core.from_dlpack not initialized"))?
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
            Framework::Tensorflow => {
                // TensorFlow doesn't support from_dlpack well, fall back
                // Need to copy the bytes since we can't hold the borrow across the call
                let bytes_vec: Vec<u8> = {
                    let buf_ref = bound_buffer.borrow();
                    buf_ref.buffer.as_slice().ok_or_else(|| {
                        SafetensorError::new_err("Buffer not on CPU")
                    })?.to_vec()
                };
                let py_bytes = PyByteArray::new(py, &bytes_vec);
                return create_tensor(
                    &self.framework,
                    dtype,
                    shape,
                    py_bytes.unbind().into_any(),
                    &self.device,
                );
            }
        };

        // View as proper dtype (from u8) and reshape
        let tensor = match self.framework {
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
                let target_dtype = get_mlx_dtype(&mlx, dtype)?;
                tensor
                    .call_method1(intern!(py, "view"), (target_dtype,))?
                    .call_method1(intern!(py, "reshape"), (shape.to_vec(),))?
            }
            Framework::Paddle => {
                let paddle = get_module(py, &PADDLE_MODULE)?;
                let target_dtype = get_paddle_dtype(&paddle, dtype)?;
                let casted = paddle.call_method1(intern!(py, "cast"), (tensor, target_dtype))?;
                paddle.call_method1(intern!(py, "reshape"), (casted, shape.to_vec()))?
            }
            Framework::Tensorflow => unreachable!(),
        };

        // Keep buffer alive
        tensor.setattr(intern!(py, "_safetensors_buffer"), py_buffer)?;

        Ok(tensor.into())
    }
}

#[pymethods]
impl PyTensorIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<(String, PyObject)>> {
        let inner = self.inner.as_mut().ok_or_else(|| {
            SafetensorError::new_err("Iterator exhausted")
        })?;

        match inner.next() {
            Some(Ok((name, buffer))) => {
                let tensor = self.buffer_to_tensor(py, &name, buffer)?;
                Ok(Some((name, tensor)))
            }
            Some(Err(e)) => Err(SafetensorError::new_err(format!("Load error: {e}"))),
            None => {
                self.inner = None;
                Ok(None)
            }
        }
    }

    fn __len__(&self) -> usize {
        self.inner.as_ref().map(|i| i.remaining()).unwrap_or(0)
    }
}

#[pyclass]
struct PySafeSlice {
    info: TensorInfo,
    framework: Framework,
    offset: usize,
    device: Device,
    loader: Arc<CoreLoader>,
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

            // Handle F4 dtype shape adjustment (PyTorch only)
            let mut shape = self.info.shape.to_vec();
            if self.framework == Framework::Pytorch && self.info.dtype == Dtype::F4 {
                let n = shape.len();
                if shape[n - 1] % 2 != 0 {
                    return Err(SafetensorError::new_err(format!(
                        "f4_x2 dtype requires that the last dim be divisible by 2 in torch: got {shape:?}",
                    )));
                }
                shape[n - 1] /= 2;
            }

            // Load full tensor to GPU
            let full_tensor = Python::with_gil(|py| {
                fetch_cuda_tensor(
                    &self.loader,
                    py,
                    start,
                    end,
                    self.info.dtype,
                    &shape,
                    &self.framework,
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

        // TODO: Currently loads full tensor then slices. Could be optimized to load only
        // the required byte ranges by: (1) computing slice ranges upfront using SliceIterator
        // logic without data, (2) adding fetchv support to hmll Rust wrapper, (3) loading
        // only the needed ranges. This would reduce I/O for sparse slice patterns.
        let start = self.info.data_offsets.0 + self.offset;
        let end = self.info.data_offsets.1 + self.offset;
        let buffer = self
            .loader
            .fetch(start, end)
            .map_err(|e| SafetensorError::new_err(format!("Loader fetch error: {e}")))?;

        let data = buffer
            .as_slice()
            .ok_or_else(|| SafetensorError::new_err("Buffer not on CPU"))?;

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

/// Create a tensor from a CpuBuffer via DLPack protocol.
///
/// Uses framework-specific from_dlpack() for potential zero-copy tensor creation.
/// The CpuBuffer exposes data as uint8 via DLPack, then we use view() + reshape()
/// for dtype reinterpretation.
///
/// The cpu_buffer is attached to the tensor to ensure memory stays alive.
fn create_tensor_from_dlpack(
    framework: &Framework,
    cpu_buffer: Py<CpuBuffer>,
    device: &Device,
    dtype: Dtype,
    shape: &[usize],
) -> PyResult<PyObject> {
    Python::with_gil(|py| -> PyResult<PyObject> {
        let buffer_obj = cpu_buffer.bind(py);

        let tensor: PyBound<'_, PyAny> = match framework {
            Framework::Pytorch => {
                // torch.from_dlpack(cpu_buffer).view(dtype).reshape(shape)
                let from_dlpack = TORCH_FROM_DLPACK
                    .get()
                    .ok_or_else(|| SafetensorError::new_err("torch.from_dlpack not initialized"))?
                    .bind(py);
                let uint8_tensor = from_dlpack.call1((buffer_obj,))?;

                let torch = get_module(py, &TORCH_MODULE)?;
                let torch_dtype = get_pydtype(torch, dtype, false)?;
                let typed_tensor =
                    uint8_tensor.call_method1(intern!(py, "view"), (torch_dtype,))?;
                let tensor =
                    typed_tensor.call_method1(intern!(py, "reshape"), (shape.to_vec(),))?;

                // Attach the CpuBuffer to keep memory alive
                tensor.setattr(intern!(py, "_safetensors_buffer"), cpu_buffer)?;

                // Move to target device if needed
                if device != &Device::Cpu {
                    let device_obj: PyObject = device.clone().into_pyobject(py)?.into();
                    let kwargs = PyDict::new(py);
                    tensor.call_method("to", (device_obj,), Some(&kwargs))?
                } else {
                    tensor
                }
            }
            Framework::Numpy => {
                // np.from_dlpack(cpu_buffer).view(dtype).reshape(shape)
                let from_dlpack = NUMPY_FROM_DLPACK
                    .get()
                    .ok_or_else(|| SafetensorError::new_err("numpy.from_dlpack not initialized"))?
                    .bind(py);
                let uint8_array = from_dlpack.call1((buffer_obj,))?;

                let numpy = get_module(py, &NUMPY_MODULE)?;
                let target_dtype = get_pydtype(numpy, dtype, true)?;
                let typed_array = uint8_array.call_method1(intern!(py, "view"), (target_dtype,))?;
                typed_array.call_method1(intern!(py, "reshape"), (shape.to_vec(),))?
                // Note: numpy keeps reference via DLPack mechanism
            }
            Framework::Flax => {
                // jax.numpy.from_dlpack(cpu_buffer).view(dtype).reshape(shape)
                let from_dlpack = JAX_FROM_DLPACK
                    .get()
                    .ok_or_else(|| {
                        SafetensorError::new_err("jax.numpy.from_dlpack not initialized")
                    })?
                    .bind(py);
                let uint8_array = from_dlpack.call1((buffer_obj,))?;

                // JAX arrays need numpy-style dtype for view
                let jax = FLAX_MODULE
                    .get()
                    .ok_or_else(|| SafetensorError::new_err("jax module not initialized"))?
                    .bind(py);
                let jax_dtype = get_jax_dtype(jax, dtype)?;
                let typed_array = uint8_array.call_method1(intern!(py, "view"), (jax_dtype,))?;
                typed_array.call_method1(intern!(py, "reshape"), (shape.to_vec(),))?
            }
            Framework::Tensorflow => {
                // TensorFlow doesn't support zero-copy view-casting, so use numpy as intermediate
                let np_from_dlpack = NUMPY_FROM_DLPACK
                    .get()
                    .ok_or_else(|| SafetensorError::new_err("numpy.from_dlpack not initialized"))?
                    .bind(py);
                let uint8_array = np_from_dlpack.call1((buffer_obj,))?;

                let numpy = get_module(py, &NUMPY_MODULE)?;
                let np_dtype = get_pydtype(numpy, dtype, true)?;
                let typed_array = uint8_array.call_method1(intern!(py, "view"), (np_dtype,))?;
                let shaped_array =
                    typed_array.call_method1(intern!(py, "reshape"), (shape.to_vec(),))?;

                // Convert numpy array to TensorFlow tensor (copies data)
                let tf = TENSORFLOW_MODULE
                    .get()
                    .ok_or_else(|| SafetensorError::new_err("tensorflow module not initialized"))?
                    .bind(py);
                tf.getattr(intern!(py, "constant"))?.call1((shaped_array,))?
            }
            Framework::Mlx => {
                // mx.from_dlpack(cpu_buffer).view(dtype).reshape(shape)
                let from_dlpack = MLX_FROM_DLPACK
                    .get()
                    .ok_or_else(|| {
                        SafetensorError::new_err("mlx.core.from_dlpack not initialized")
                    })?
                    .bind(py);
                let uint8_array = from_dlpack.call1((buffer_obj,))?;

                let mlx = MLX_MODULE
                    .get()
                    .ok_or_else(|| SafetensorError::new_err("mlx module not initialized"))?
                    .bind(py);
                let mlx_dtype = get_mlx_dtype(mlx, dtype)?;
                let typed_array = uint8_array.call_method1(intern!(py, "view"), (mlx_dtype,))?;
                typed_array.call_method1(intern!(py, "reshape"), (shape.to_vec(),))?
            }
            Framework::Paddle => {
                // paddle.from_dlpack(cpu_buffer).view(dtype).reshape(shape)
                let from_dlpack = PADDLE_FROM_DLPACK
                    .get()
                    .ok_or_else(|| SafetensorError::new_err("paddle.from_dlpack not initialized"))?
                    .bind(py);
                let uint8_tensor = from_dlpack.call1((buffer_obj,))?;

                let paddle = PADDLE_MODULE
                    .get()
                    .ok_or_else(|| SafetensorError::new_err("paddle module not initialized"))?
                    .bind(py);
                let paddle_dtype = get_paddle_dtype(paddle, dtype)?;

                // Paddle uses cast instead of view for dtype conversion
                let typed_tensor =
                    uint8_tensor.call_method1(intern!(py, "cast"), (paddle_dtype,))?;
                let tensor =
                    typed_tensor.call_method1(intern!(py, "reshape"), (shape.to_vec(),))?;

                // Move to target device if needed
                if let Device::Cuda(index) = device {
                    let device_str = format!("gpu:{index}");
                    tensor.call_method1(intern!(py, "cuda"), (device_str,))?
                } else {
                    tensor
                }
            }
        };
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
        f: PyObject,
        framework: Framework,
        device: Option<Device>,
        backend: Option<Backend>,
    ) -> PyResult<Self> {
        let filename = Python::with_gil(|py| -> PyResult<PathBuf> {
            let _ = f.getattr(py, "fileno")?;
            let filename = f.getattr(py, "name")?;
            let filename: PathBuf = filename.extract(py)?;
            Ok(filename)
        })?;
        let inner = Some(Open::new(filename, framework, device, backend)?);
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
    m.add_class::<PyTensorIterator>()?;
    m.add("SafetensorError", m.py().get_type::<SafetensorError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
