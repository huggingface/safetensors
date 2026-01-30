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
use safetensors::loader::{Buffer as LoaderBuffer, Device as LoaderDevice, Loader as CoreLoader};
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
static TORCH_AS_TENSOR: OnceLock<Py<PyAny>> = OnceLock::new();
static TORCH_FROM_DLPACK: OnceLock<Py<PyAny>> = OnceLock::new();
static NUMPY_MODULE: OnceLock<Py<PyModule>> = OnceLock::new();
static NUMPY_ASARRAY: OnceLock<Py<PyAny>> = OnceLock::new();
static TENSORFLOW_MODULE: OnceLock<Py<PyModule>> = OnceLock::new();
static FLAX_MODULE: OnceLock<Py<PyModule>> = OnceLock::new();
static MLX_MODULE: OnceLock<Py<PyModule>> = OnceLock::new();
static PADDLE_MODULE: OnceLock<Py<PyModule>> = OnceLock::new();

/// A CPU buffer wrapper that exposes `__array_interface__` for zero-copy
/// tensor creation with NumPy, PyTorch, and other frameworks.
///
/// Always exposes data as uint8 - callers use view() + reshape() for dtype conversion.
/// This simplifies the code path and works uniformly for all dtypes including exotic ones.
#[pyclass]
struct CpuBuffer {
    /// The underlying buffer (kept alive as long as this object exists)
    buffer: LoaderBuffer,
    /// Size in bytes (exposed as 1D uint8 array)
    nbytes: usize,
}

impl CpuBuffer {
    /// Create a new CpuBuffer from a loader Buffer.
    fn new(buffer: LoaderBuffer) -> Self {
        let nbytes = buffer.len();
        Self { buffer, nbytes }
    }
}

#[pymethods]
impl CpuBuffer {
    /// Returns the NumPy array interface dict for zero-copy interop.
    ///
    /// Always returns uint8 dtype - callers use view() + reshape() for conversion.
    /// See: https://numpy.org/doc/stable/reference/arrays.interface.html
    #[getter]
    fn __array_interface__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);

        // Shape as 1D array of bytes
        let shape_tuple = pyo3::types::PyTuple::new(py, &[self.nbytes])?;
        dict.set_item("shape", shape_tuple)?;

        // Always uint8
        dict.set_item("typestr", "|u1")?;

        // Data pointer and read-only flag: (ptr, readonly)
        let ptr = self.buffer.as_ptr() as usize;
        let data_tuple = (ptr, false).into_pyobject(py)?;
        dict.set_item("data", data_tuple)?;

        // Version (3 is the current version)
        dict.set_item("version", 3)?;

        // Optional: strides (None means C-contiguous)
        dict.set_item("strides", py.None())?;

        Ok(dict.into())
    }

    /// Returns the size in bytes.
    #[getter]
    fn nbytes(&self) -> usize {
        self.nbytes
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
        // For F8 and F4 types, export as uint8 and view-cast
        _ => return None,
    })
}

/// High-performance loader for safetensors files.
///
/// This is a thin wrapper around the core Loader that adds Python-specific
/// functionality like CUDA tensor creation via DLPack protocol.
struct Loader {
    /// The core loader (handles file access, device targeting, synchronization)
    core: CoreLoader,
    /// CUDA device index (if applicable)
    cuda_index: Option<usize>,
}

impl Loader {
    /// Create a new Loader from a file path with specified device.
    fn new(path: &std::path::Path, device: LoaderDevice) -> Result<Self, safetensors::loader::LoaderError> {
        let cuda_index = match device {
            LoaderDevice::Cpu => None,
            LoaderDevice::Cuda(idx) => Some(idx),
        };
        let core = CoreLoader::open(path, device)?;
        Ok(Self { core, cuda_index })
    }

    /// Fetch a range of bytes and return the raw Buffer.
    /// For CUDA, this buffer is on the GPU.
    fn fetch_buffer(&self, start: usize, end: usize) -> Result<LoaderBuffer, safetensors::loader::LoaderError> {
        self.core.fetch(start, end)
    }

    /// Fetch a range of bytes and create a PyTorch CUDA tensor with ZERO COPY.
    ///
    /// This loads data directly to GPU, then creates a PyTorch tensor
    /// that wraps the GPU memory via DLPack protocol (6x faster than
    /// __cuda_array_interface__). For exotic dtypes (F8, F4), exports
    /// as uint8 via DLPack then view-casts to the target dtype.
    ///
    /// A GpuBufferHolder is attached to the tensor to keep the GPU memory alive.
    fn fetch_cuda_tensor(
        &self,
        py: Python<'_>,
        start: usize,
        end: usize,
        dtype: Dtype,
        shape: &[usize],
    ) -> PyResult<PyObject> {
        // Ensure we're configured for CUDA
        if !matches!(self.core.device(), LoaderDevice::Cuda(_)) {
            return Err(SafetensorError::new_err(
                "Loader not configured for CUDA",
            ));
        }

        let device_index = self.cuda_index.unwrap_or(0);

        // Note: CUDA device is set once when the loader is created (in safe_open/SafeLoader::new)
        // This avoids the overhead of calling torch.cuda.set_device() for every tensor fetch

        // Fetch data directly to GPU
        let buffer = self.fetch_buffer(start, end).map_err(|e| {
            SafetensorError::new_err(format!("Loader fetch error: {e}"))
        })?;

        let nbytes = buffer.len();
        let ptr = buffer.as_ptr() as *mut std::ffi::c_void;

        // For exotic dtypes, export as uint8 (caller will view-cast)
        let (dlpack_dtype, dlpack_shape) = if let Some(dt) = dtype_to_dlpack(dtype) {
            let shape_i64: Vec<i64> = shape.iter().map(|&s| s as i64).collect();
            (dt, shape_i64)
        } else {
            // Export as flat uint8 array for exotic dtypes
            (dlpack_ffi::DataType::U8, vec![nbytes as i64])
        };

        // Create DLPack wrapper and capsule directly (faster than going through __dlpack__)
        let wrapper = DlpackGpuWrapper {
            ptr,
            shape: dlpack_shape,
            dtype: dlpack_dtype,
            device_index,
        };

        let managed_tensor = SafeManagedTensor::new(wrapper)
            .map_err(|e| SafetensorError::new_err(format!("Failed to create DLPack tensor: {:?}", e)))?;

        let capsule = managed_tensor_to_capsule(py, managed_tensor)?;

        // Get cached torch.from_dlpack
        let from_dlpack = TORCH_FROM_DLPACK
            .get()
            .ok_or_else(|| SafetensorError::new_err("torch.from_dlpack not initialized"))?
            .bind(py);

        // Create tensor from DLPack capsule
        let tensor = from_dlpack.call1((capsule,))?;

        // For exotic dtypes, view-cast from uint8 to target dtype and reshape
        let tensor = if dtype_to_dlpack(dtype).is_none() {
            let torch = get_module(py, &TORCH_MODULE)?;
            let target_dtype = get_pydtype(torch, dtype, false)?;
            let typed = tensor.call_method1(intern!(py, "view"), (target_dtype,))?;
            typed.call_method1(intern!(py, "reshape"), (shape.to_vec(),))?
        } else {
            tensor
        };

        // Create lightweight holder to keep buffer alive and attach to tensor
        let holder = Py::new(py, GpuBufferHolder { buffer })?;
        tensor.setattr(intern!(py, "_safetensors_buffer"), holder)?;

        Ok(tensor.into())
    }

    /// Fetch a range of bytes and create a CPU tensor using zero-copy.
    ///
    /// Always creates a uint8 array via __array_interface__, then uses view() + reshape()
    /// for zero-copy dtype reinterpretation. This unified path works for all dtypes.
    /// For zero-sized tensors, falls back to copy-based approach (null pointers).
    ///
    /// Uses fetch_view() for true zero-copy mmap access when available.
    fn fetch_cpu_tensor(
        &self,
        py: Python<'_>,
        start: usize,
        end: usize,
        dtype: Dtype,
        shape: &[usize],
        framework: &Framework,
        device: &Device,
    ) -> PyResult<PyObject> {
        // Check if tensor is zero-sized (any dimension is 0)
        let is_zero_sized = shape.iter().any(|&d| d == 0);

        if is_zero_sized {
            // Zero-sized tensors have null pointers - use copy-based path
            let data = self.core.fetch_to_vec(start, end).map_err(|e| {
                SafetensorError::new_err(format!("Loader fetch error: {e}"))
            })?;
            let array: PyObject = PyByteArray::new(py, &data).into_any().into();
            return create_tensor(framework, dtype, shape, array, device);
        }

        // Try zero-copy fetch_view first (true mmap zero-copy)
        // Falls back to fetch() with copy if view is not available
        let buffer = self.core.fetch_view(start, end).or_else(|_| {
            // fetch_view not available (e.g., non-mmap backend), fall back to fetch
            self.fetch_buffer(start, end)
        }).map_err(|e| {
            SafetensorError::new_err(format!("Loader fetch error: {e}"))
        })?;

        let asarray = NUMPY_ASARRAY
            .get()
            .ok_or_else(|| SafetensorError::new_err("numpy module not initialized"))?
            .bind(py);

        // Unified path: uint8 array via __array_interface__, then view() + reshape()
        let cpu_buffer = Py::new(py, CpuBuffer::new(buffer))?;
        let raw_array = asarray.call1((cpu_buffer.bind(py),))?;

        // Pass to framework-specific handler with dtype info for view-casting
        create_tensor_from_numpy(framework, &raw_array, device, cpu_buffer, dtype, shape)
    }

    /// Get the loader device.
    fn device(&self) -> LoaderDevice {
        self.core.device()
    }

    /// Get the CUDA device index (if applicable).
    #[allow(dead_code)]
    fn cuda_index(&self) -> Option<usize> {
        self.cuda_index
    }
}

/// Convert Python Device to loader Device.
/// PyTorch CUDA devices get direct GPU loading via __cuda_array_interface__.
/// For other frameworks/devices, use CPU and let the framework handle device transfer.
fn device_to_loader(device: &Device, framework: &Framework) -> LoaderDevice {
    match (device, framework) {
        // PyTorch CUDA devices get direct GPU loading
        // hmll uses the current CUDA context, so we'll set the device before operations
        (Device::Cuda(idx), Framework::Pytorch) => LoaderDevice::Cuda(*idx),
        _ => LoaderDevice::Cpu,
    }
}

/// Create a Loader with the appropriate device configuration.
fn create_loader(
    filename: &std::path::Path,
    device: &Device,
    framework: &Framework,
) -> Result<Loader, safetensors::loader::LoaderError> {
    let loader_device = device_to_loader(device, framework);
    Loader::new(filename, loader_device)
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

struct Open {
    metadata: Metadata,
    offset: usize,
    framework: Framework,
    device: Device,
    loader: Arc<Loader>,
}

impl Open {
    fn new(filename: PathBuf, framework: Framework, device: Option<Device>) -> PyResult<Self> {
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
            // Always initialize numpy - needed for zero-copy path via __array_interface__
            let numpy = PyModule::import(py, intern!(py, "numpy"))?;
            // Cache numpy.asarray for faster access
            let asarray = numpy.getattr(intern!(py, "asarray"))?;
            NUMPY_ASARRAY.get_or_init_py_attached(py, || asarray.unbind());
            NUMPY_MODULE.get_or_init_py_attached(py, || numpy.into());

            // Also initialize framework-specific modules
            match framework {
                Framework::Pytorch => {
                    let module = PyModule::import(py, intern!(py, "torch"))?;
                    // Cache torch.as_tensor for faster access (avoids getattr per tensor)
                    let as_tensor = module.getattr(intern!(py, "as_tensor"))?;
                    TORCH_AS_TENSOR.get_or_init_py_attached(py, || as_tensor.unbind());
                    // Cache torch.from_dlpack for DLPack-based tensor creation (6x faster)
                    let from_dlpack = module.getattr(intern!(py, "from_dlpack"))?;
                    TORCH_FROM_DLPACK.get_or_init_py_attached(py, || from_dlpack.unbind());
                    // Set CUDA device once here BEFORE storing the module
                    // hmll uses the current CUDA context, so this is much more
                    // efficient than setting it per-tensor fetch
                    if let Device::Cuda(idx) = device {
                        module
                            .getattr(intern!(py, "cuda"))?
                            .getattr(intern!(py, "set_device"))?
                            .call1((idx,))?;
                    }
                    TORCH_MODULE.get_or_init_py_attached(py, || module.into());
                }
                Framework::Paddle => {
                    let module = PyModule::import(py, intern!(py, "paddle"))?;
                    PADDLE_MODULE.get_or_init_py_attached(py, || module.into());
                }
                _ => {} // numpy already initialized above
            };

            Ok(())
        })?;

        // Create high-performance loader (CUDA device already set above if needed)
        let loader = create_loader(&filename, &device, &framework).map_err(|e| {
            SafetensorError::new_err(format!("Failed to create loader: {e}"))
        })?;

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

        // For PyTorch + CUDA loader, use zero-copy path via __cuda_array_interface__
        // NOTE: We check the loader's device, not self.device, because the loader
        // may be configured for CPU even when target device is CUDA (e.g., cuda:1
        // falls back to CPU loader since hmll only supports cuda:0 direct loading)
        if self.framework == Framework::Pytorch
            && matches!(self.loader.device(), LoaderDevice::Cuda(_))
        {
            return self.loader
                .fetch_cuda_tensor(py, start, end, info.dtype, &shape);
        }

        // CPU path: use zero-copy fetch_cpu_tensor
        self.loader.fetch_cpu_tensor(
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
    #[pyo3(signature = (filename, framework, device=Some(Device::Cpu)))]
    fn new(filename: PathBuf, framework: Framework, device: Option<Device>) -> PyResult<Self> {
        let inner = Some(Open::new(filename, framework, device)?);
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

#[pyclass]
struct PySafeSlice {
    info: TensorInfo,
    framework: Framework,
    offset: usize,
    device: Device,
    loader: Arc<Loader>,
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
        // For PyTorch + CUDA loader, load full tensor to GPU then slice on GPU
        // (CPU-side slicing requires CPU buffer which CUDA loader can't provide)
        // NOTE: We check the loader's device, not self.device, because the loader
        // may be configured for CPU even when target device is CUDA
        if self.framework == Framework::Pytorch
            && matches!(self.loader.device(), LoaderDevice::Cuda(_))
        {
            let start = self.info.data_offsets.0 + self.offset;
            let end = self.info.data_offsets.1 + self.offset;

            // Handle F4 dtype shape adjustment
            let mut shape = self.info.shape.to_vec();
            if self.info.dtype == Dtype::F4 {
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
                self.loader
                    .fetch_cuda_tensor(py, start, end, self.info.dtype, &shape)
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

        // Fetch the tensor data using loader (zero-copy via buffer protocol)
        let start = self.info.data_offsets.0 + self.offset;
        let end = self.info.data_offsets.1 + self.offset;
        let buffer = self.loader.fetch_buffer(start, end).map_err(|e| {
            SafetensorError::new_err(format!("Loader fetch error: {e}"))
        })?;

        let data = buffer
            .as_slice()
            .ok_or_else(|| SafetensorError::new_err("Buffer not on CPU"))?;

        let shape = self.info.shape.clone();

        let tensor = TensorView::new(self.info.dtype, self.info.shape.clone(), data).map_err(
            |e| SafetensorError::new_err(format!("Error preparing tensor view: {e}")),
        )?;
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

fn create_tensor<'a>(
    framework: &'a Framework,
    dtype: Dtype,
    shape: &'a [usize],
    array: PyObject,
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

/// Create a tensor from a uint8 numpy array using zero-copy view + reshape.
///
/// The array is always uint8 (raw bytes from CpuBuffer). We use view() + reshape()
/// for zero-copy dtype reinterpretation. This unified path works for all dtypes.
///
/// The cpu_buffer is attached to the tensor to ensure memory stays alive.
fn create_tensor_from_numpy(
    framework: &Framework,
    array: &PyBound<'_, PyAny>,
    device: &Device,
    cpu_buffer: Py<CpuBuffer>,
    dtype: Dtype,
    shape: &[usize],
) -> PyResult<PyObject> {
    Python::with_gil(|py| -> PyResult<PyObject> {
        let tensor: PyBound<'_, PyAny> = match framework {
            Framework::Pytorch => {
                let torch = get_module(py, &TORCH_MODULE)?;

                // torch.from_numpy(uint8_array).view(torch_dtype).reshape(shape)
                let uint8_tensor = torch.getattr(intern!(py, "from_numpy"))?.call1((array,))?;
                let torch_dtype = get_pydtype(torch, dtype, false)?;
                let typed_tensor = uint8_tensor.call_method1(intern!(py, "view"), (torch_dtype,))?;
                let tensor = typed_tensor.call_method1(intern!(py, "reshape"), (shape.to_vec(),))?;

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
                let numpy = get_module(py, &NUMPY_MODULE)?;
                // numpy view() + reshape() is zero-copy
                let target_dtype = get_pydtype(numpy, dtype, true)?;
                let typed_array = array.call_method1(intern!(py, "view"), (target_dtype,))?;
                typed_array.call_method1(intern!(py, "reshape"), (shape.to_vec(),))?
                // Note: numpy keeps reference via array.base chain
            }
            Framework::Flax => {
                let numpy = get_module(py, &NUMPY_MODULE)?;
                // First view-cast in numpy, then convert to JAX (which copies)
                let target_dtype = get_pydtype(numpy, dtype, true)?;
                let typed_array = array.call_method1(intern!(py, "view"), (target_dtype,))?;
                let shaped_array = typed_array.call_method1(intern!(py, "reshape"), (shape.to_vec(),))?;

                let module = Python::with_gil(|py| -> PyResult<&Py<PyModule>> {
                    let module = PyModule::import(py, intern!(py, "jax"))?;
                    Ok(FLAX_MODULE.get_or_init_py_attached(py, || module.into()))
                })?
                .bind(py);
                module
                    .getattr(intern!(py, "numpy"))?
                    .getattr(intern!(py, "array"))?
                    .call1((shaped_array,))?
            }
            Framework::Tensorflow => {
                let numpy = get_module(py, &NUMPY_MODULE)?;
                // First view-cast in numpy, then convert to TensorFlow (which copies)
                let target_dtype = get_pydtype(numpy, dtype, true)?;
                let typed_array = array.call_method1(intern!(py, "view"), (target_dtype,))?;
                let shaped_array = typed_array.call_method1(intern!(py, "reshape"), (shape.to_vec(),))?;

                let module = Python::with_gil(|py| -> PyResult<&Py<PyModule>> {
                    let module = PyModule::import(py, intern!(py, "tensorflow"))?;
                    Ok(TENSORFLOW_MODULE.get_or_init_py_attached(py, || module.into()))
                })?
                .bind(py);
                module
                    .getattr(intern!(py, "convert_to_tensor"))?
                    .call1((shaped_array,))?
            }
            Framework::Mlx => {
                let numpy = get_module(py, &NUMPY_MODULE)?;
                // First view-cast in numpy, then convert to MLX (which copies)
                let target_dtype = get_pydtype(numpy, dtype, true)?;
                let typed_array = array.call_method1(intern!(py, "view"), (target_dtype,))?;
                let shaped_array = typed_array.call_method1(intern!(py, "reshape"), (shape.to_vec(),))?;

                let module = Python::with_gil(|py| -> PyResult<&Py<PyModule>> {
                    let module = PyModule::import(py, intern!(py, "mlx"))?;
                    Ok(MLX_MODULE.get_or_init_py_attached(py, || module.into()))
                })?
                .bind(py);
                module
                    .getattr(intern!(py, "core"))?
                    .call_method1("array", (shaped_array,))?
            }
            Framework::Paddle => {
                let numpy = get_module(py, &NUMPY_MODULE)?;
                // First view-cast in numpy, then convert to Paddle
                let target_dtype = get_pydtype(numpy, dtype, true)?;
                let typed_array = array.call_method1(intern!(py, "view"), (target_dtype,))?;
                let shaped_array = typed_array.call_method1(intern!(py, "reshape"), (shape.to_vec(),))?;

                let module = Python::with_gil(|py| -> PyResult<&Py<PyModule>> {
                    let module = PyModule::import(py, intern!(py, "paddle"))?;
                    Ok(PADDLE_MODULE.get_or_init_py_attached(py, || module.into()))
                })?
                .bind(py);
                let device_obj: PyObject = if let Device::Cuda(index) = device {
                    format!("gpu:{index}").into_pyobject(py)?.into()
                } else {
                    device.clone().into_pyobject(py)?.into()
                };
                let kwargs = [(intern!(py, "place"), device_obj)].into_py_dict(py)?;
                module
                    .getattr(intern!(py, "to_tensor"))?
                    .call((shaped_array,), Some(&kwargs))?
            }
        };
        Ok(tensor.into())
    })
}

fn get_pydtype(module: &PyBound<'_, PyModule>, dtype: Dtype, is_numpy: bool) -> PyResult<PyObject> {
    Python::with_gil(|py| {
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
    })
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
    #[pyo3(signature = (f, framework, device=Some(Device::Cpu)))]
    fn new(f: PyObject, framework: Framework, device: Option<Device>) -> PyResult<Self> {
        let filename = Python::with_gil(|py| -> PyResult<PathBuf> {
            let _ = f.getattr(py, "fileno")?;
            let filename = f.getattr(py, "name")?;
            let filename: PathBuf = filename.extract(py)?;
            Ok(filename)
        })?;
        let inner = Some(Open::new(filename, framework, device)?);
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
    m.add("SafetensorError", m.py().get_type::<SafetensorError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_parse() {
        let torch_version = "1.1.1";
        let version = Version::from_string(torch_version).unwrap();
        assert_eq!(version, Version::new(1, 1, 1));

        let torch_version = "2.0.0a0+gitd1123c9";
        let version = Version::from_string(torch_version).unwrap();
        assert_eq!(version, Version::new(2, 0, 0));

        let torch_version = "something";
        let version = Version::from_string(torch_version);
        assert!(version.is_err());
    }
}
