#![deny(missing_docs)]
//! Dummy doc
use memmap2::{Mmap, MmapOptions};
use pyo3::exceptions::{PyException, PyFileNotFoundError};
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::IntoPyDict;
use pyo3::types::PySlice;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyList};
use pyo3::Bound as PyBound;
use pyo3::{intern, PyErr};
use safetensors::slice::TensorIndexer;
use safetensors::tensor::{Dtype, Metadata, SafeTensors, TensorInfo, TensorView};
use safetensors::View;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::iter::FromIterator;
use std::ops::Bound;
use std::path::PathBuf;
use std::sync::Arc;

static TORCH_MODULE: GILOnceCell<Py<PyModule>> = GILOnceCell::new();
static NUMPY_MODULE: GILOnceCell<Py<PyModule>> = GILOnceCell::new();
static TENSORFLOW_MODULE: GILOnceCell<Py<PyModule>> = GILOnceCell::new();
static FLAX_MODULE: GILOnceCell<Py<PyModule>> = GILOnceCell::new();
static MLX_MODULE: GILOnceCell<Py<PyModule>> = GILOnceCell::new();

struct PyView<'a> {
    shape: Vec<usize>,
    dtype: Dtype,
    data: PyBound<'a, PyBytes>,
    data_len: usize,
}

impl<'a> View for &PyView<'a> {
    fn data(&self) -> std::borrow::Cow<[u8]> {
        Cow::Borrowed(self.data.as_bytes())
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn dtype(&self) -> Dtype {
        self.dtype
    }
    fn data_len(&self) -> usize {
        self.data_len
    }
}

fn prepare(tensor_dict: HashMap<String, PyBound<PyDict>>) -> PyResult<HashMap<String, PyView>> {
    let mut tensors = HashMap::with_capacity(tensor_dict.len());
    for (tensor_name, tensor_desc) in &tensor_dict {
        let shape: Vec<usize> = tensor_desc
            .get_item("shape")?
            .ok_or_else(|| SafetensorError::new_err(format!("Missing `shape` in {tensor_desc:?}")))?
            .extract()?;
        let pydata: PyBound<PyAny> = tensor_desc.get_item("data")?.ok_or_else(|| {
            SafetensorError::new_err(format!("Missing `data` in {tensor_desc:?}"))
        })?;
        // Make sure it's extractable first.
        let data: &[u8] = pydata.extract()?;
        let data_len = data.len();
        let data: PyBound<PyBytes> = pydata.extract()?;
        let pydtype = tensor_desc.get_item("dtype")?.ok_or_else(|| {
            SafetensorError::new_err(format!("Missing `dtype` in {tensor_desc:?}"))
        })?;
        let dtype: &str = pydtype.extract()?;
        let dtype = match dtype {
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
            dtype_str => {
                return Err(SafetensorError::new_err(format!(
                    "dtype {dtype_str} is not covered",
                )));
            }
        };

        let tensor = PyView {
            shape,
            dtype,
            data,
            data_len,
        };
        tensors.insert(tensor_name.to_string(), tensor);
    }
    Ok(tensors)
}

/// Serializes raw data.
///
/// Args:
///     tensor_dict (`Dict[str, Dict[Any]]`):
///         The tensor dict is like:
///             {"tensor_name": {"dtype": "F32", "shape": [2, 3], "data": b"\0\0"}}
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
    let tensors = prepare(tensor_dict)?;
    let metadata_map = metadata.map(HashMap::from_iter);
    let out = safetensors::tensor::serialize(&tensors, &metadata_map)
        .map_err(|e| SafetensorError::new_err(format!("Error while serializing: {e:?}")))?;
    let pybytes = PyBytes::new_bound(py, &out);
    Ok(pybytes)
}

/// Serializes raw data.
///
/// Args:
///     tensor_dict (`Dict[str, Dict[Any]]`):
///         The tensor dict is like:
///             {"tensor_name": {"dtype": "F32", "shape": [2, 3], "data": b"\0\0"}}
///     filename (`str`, or `os.PathLike`):
///         The name of the file to write into.
///     metadata (`Dict[str, str]`, *optional*):
///         The optional purely text annotations
///
/// Returns:
///     (`bytes`):
///         The serialized content.
#[pyfunction]
#[pyo3(signature = (tensor_dict, filename, metadata=None))]
fn serialize_file(
    tensor_dict: HashMap<String, PyBound<PyDict>>,
    filename: PathBuf,
    metadata: Option<HashMap<String, String>>,
) -> PyResult<()> {
    let tensors = prepare(tensor_dict)?;
    safetensors::tensor::serialize_to_file(&tensors, &metadata, filename.as_path())
        .map_err(|e| SafetensorError::new_err(format!("Error while serializing: {e:?}")))?;
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
        .map_err(|e| SafetensorError::new_err(format!("Error while deserializing: {e:?}")))?;

    let tensors = safetensor.tensors();
    let mut items = Vec::with_capacity(tensors.len());

    for (tensor_name, tensor) in tensors {
        let pyshape: PyObject = PyList::new_bound(py, tensor.shape().iter()).into();
        let pydtype: PyObject = format!("{:?}", tensor.dtype()).into_py(py);

        let pydata: PyObject = PyByteArray::new_bound(py, tensor.data()).into();

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
                let idx = dim
                    .checked_add_signed(idx as isize)
                    .ok_or(SafetensorError::new_err(format!(
                        "Invalid index {idx} for dimension {dim_idx} of size {dim}"
                    )))?;
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
}

impl<'source> FromPyObject<'source> for Device {
    fn extract_bound(ob: &PyBound<'source, PyAny>) -> PyResult<Self> {
        if let Ok(name) = ob.extract::<String>() {
            match &name[..] {
                "cpu" => Ok(Device::Cpu),
                "cuda" => Ok(Device::Cuda(0)),
                "mps" => Ok(Device::Mps),
                "npu" => Ok(Device::Npu(0)),
                "xpu" => Ok(Device::Xpu(0)),
                name if name.starts_with("cuda:") => {
                    let tokens: Vec<_> = name.split(':').collect();
                    if tokens.len() == 2 {
                        let device: usize = tokens[1].parse()?;
                        Ok(Device::Cuda(device))
                    } else {
                        Err(SafetensorError::new_err(format!(
                            "device {name} is invalid"
                        )))
                    }
                }
                name if name.starts_with("npu:") => {
                    let tokens: Vec<_> = name.split(':').collect();
                    if tokens.len() == 2 {
                        let device: usize = tokens[1].parse()?;
                        Ok(Device::Npu(device))
                    } else {
                        Err(SafetensorError::new_err(format!(
                            "device {name} is invalid"
                        )))
                    }
                }
                name if name.starts_with("xpu:") => {
                    let tokens: Vec<_> = name.split(':').collect();
                    if tokens.len() == 2 {
                        let device: usize = tokens[1].parse()?;
                        Ok(Device::Xpu(device))
                    } else {
                        Err(SafetensorError::new_err(format!(
                            "device {name} is invalid"
                        )))
                    }
                }
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

impl IntoPy<PyObject> for Device {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            Device::Cpu => "cpu".into_py(py),
            Device::Cuda(n) => format!("cuda:{n}").into_py(py),
            Device::Mps => "mps".into_py(py),
            Device::Npu(n) => format!("npu:{n}").into_py(py),
            Device::Xpu(n) => format!("xpu:{n}").into_py(py),
        }
    }
}

enum Storage {
    Mmap(Mmap),
    /// Torch specific mmap
    /// This allows us to not manage it
    /// so Pytorch can handle the whole lifecycle.
    /// https://pytorch.org/docs/stable/storage.html#torch.TypedStorage.from_file.
    TorchStorage(GILOnceCell<PyObject>),
}

#[derive(Debug, PartialEq, Eq, PartialOrd)]
struct Version {
    major: u8,
    minor: u8,
    patch: u8,
}

impl Version {
    fn new(major: u8, minor: u8, patch: u8) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    fn from_string(string: &str) -> Result<Self, String> {
        let mut parts = string.split('.');
        let err = || format!("Could not parse torch package version {string}.");
        let major_str = parts.next().ok_or_else(err)?;
        let minor_str = parts.next().ok_or_else(err)?;
        let patch_str = parts.next().ok_or_else(err)?;
        // Patch is more complex and can be:
        // - `1` a number
        // - `1a0`, `1b0`, `1rc1` an alpha, beta, release candidate version
        // - `1a0+git2323` from source with commit number
        let patch_str: String = patch_str
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();

        let major = major_str.parse().map_err(|_| err())?;
        let minor = minor_str.parse().map_err(|_| err())?;
        let patch = patch_str.parse().map_err(|_| err())?;
        Ok(Version {
            major,
            minor,
            patch,
        })
    }
}

struct Open {
    metadata: Metadata,
    offset: usize,
    framework: Framework,
    device: Device,
    storage: Arc<Storage>,
}

impl Open {
    fn new(filename: PathBuf, framework: Framework, device: Option<Device>) -> PyResult<Self> {
        let file = File::open(&filename).map_err(|_| {
            PyFileNotFoundError::new_err(format!("No such file or directory: {filename:?}"))
        })?;
        let device = device.unwrap_or(Device::Cpu);

        if device != Device::Cpu && framework != Framework::Pytorch {
            return Err(SafetensorError::new_err(format!(
                "Device {device:?} is not support for framework {framework:?}",
            )));
        }

        // SAFETY: Mmap is used to prevent allocating in Rust
        // before making a copy within Python.
        let buffer = unsafe { MmapOptions::new().map(&file)? };

        let (n, metadata) = SafeTensors::read_metadata(&buffer).map_err(|e| {
            SafetensorError::new_err(format!("Error while deserializing header: {e:?}"))
        })?;

        let offset = n + 8;

        Python::with_gil(|py| -> PyResult<()> {
            match framework {
                Framework::Pytorch => {
                    let module = PyModule::import_bound(py, intern!(py, "torch"))?;
                    TORCH_MODULE.get_or_init(py, || module.into())
                }
                _ => {
                    let module = PyModule::import_bound(py, intern!(py, "numpy"))?;
                    NUMPY_MODULE.get_or_init(py, || module.into())
                }
            };

            Ok(())
        })?;

        let storage = match &framework {
            Framework::Pytorch => Python::with_gil(|py| -> PyResult<Storage> {
                let module = get_module(py, &TORCH_MODULE)?;

                let version: String = module.getattr(intern!(py, "__version__"))?.extract()?;
                let version = Version::from_string(&version).map_err(SafetensorError::new_err)?;

                // Untyped storage only exists for versions over 1.11.0
                // Same for torch.asarray which is necessary for zero-copy tensor
                if version >= Version::new(1, 11, 0) {
                    // storage = torch.ByteStorage.from_file(filename, shared=False, size=size).untyped()
                    let py_filename: PyObject = filename.into_py(py);
                    let size: PyObject = buffer.len().into_py(py);
                    let shared: PyObject = false.into_py(py);
                    let (size_name, storage_name) = if version >= Version::new(2, 0, 0) {
                        (intern!(py, "nbytes"), intern!(py, "UntypedStorage"))
                    } else {
                        (intern!(py, "size"), intern!(py, "ByteStorage"))
                    };

                    let kwargs =
                        [(intern!(py, "shared"), shared), (size_name, size)].into_py_dict_bound(py);
                    let storage = module
                        .getattr(storage_name)?
                        // .getattr(intern!(py, "from_file"))?
                        .call_method("from_file", (py_filename,), Some(&kwargs))?;

                    let untyped: PyBound<'_, PyAny> = match storage.getattr(intern!(py, "untyped"))
                    {
                        Ok(untyped) => untyped,
                        Err(_) => storage.getattr(intern!(py, "_untyped"))?,
                    };
                    let storage = untyped.call0()?.into_py(py);
                    let gil_storage = GILOnceCell::new();
                    gil_storage.get_or_init(py, || storage);

                    Ok(Storage::TorchStorage(gil_storage))
                } else {
                    Ok(Storage::Mmap(buffer))
                }
            })?,
            _ => Storage::Mmap(buffer),
        };

        let storage = Arc::new(storage);

        Ok(Self {
            metadata,
            offset,
            framework,
            device,
            storage,
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
    pub fn get_tensor(&self, name: &str) -> PyResult<PyObject> {
        let info = self.metadata.info(name).ok_or_else(|| {
            SafetensorError::new_err(format!("File does not contain tensor {name}",))
        })?;
        // let info = tensors.get(name).ok_or_else(|| {
        //     SafetensorError::new_err(format!("File does not contain tensor {name}",))
        // })?;

        match &self.storage.as_ref() {
            Storage::Mmap(mmap) => {
                let data =
                    &mmap[info.data_offsets.0 + self.offset..info.data_offsets.1 + self.offset];

                let array: PyObject =
                    Python::with_gil(|py| PyByteArray::new_bound(py, data).into_py(py));

                create_tensor(
                    &self.framework,
                    info.dtype,
                    &info.shape,
                    array,
                    &self.device,
                )
            }
            Storage::TorchStorage(storage) => {
                Python::with_gil(|py| -> PyResult<PyObject> {
                    let torch = get_module(py, &TORCH_MODULE)?;
                    let dtype: PyObject = get_pydtype(torch, info.dtype, false)?;
                    let torch_uint8: PyObject = get_pydtype(torch, Dtype::U8, false)?;
                    let kwargs = [(intern!(py, "dtype"), torch_uint8)].into_py_dict_bound(py);
                    let view_kwargs = [(intern!(py, "dtype"), dtype)].into_py_dict_bound(py);
                    let shape = info.shape.to_vec();
                    let shape: PyObject = shape.into_py(py);

                    let start = (info.data_offsets.0 + self.offset) as isize;
                    let stop = (info.data_offsets.1 + self.offset) as isize;
                    let slice = PySlice::new_bound(py, start, stop, 1);
                    let storage: &PyObject = storage
                        .get(py)
                        .ok_or_else(|| SafetensorError::new_err("Could not find storage"))?;
                    let storage: &PyBound<PyAny> = storage.bind(py);
                    let storage_slice = storage
                        .getattr(intern!(py, "__getitem__"))?
                        .call1((slice,))?;

                    let sys = PyModule::import_bound(py, intern!(py, "sys"))?;
                    let byteorder: String = sys.getattr(intern!(py, "byteorder"))?.extract()?;

                    let mut tensor = torch
                        .getattr(intern!(py, "asarray"))?
                        .call((storage_slice,), Some(&kwargs))?
                        .getattr(intern!(py, "view"))?
                        .call((), Some(&view_kwargs))?;

                    if byteorder == "big" {
                        let version: String =
                            torch.getattr(intern!(py, "__version__"))?.extract()?;
                        let version =
                            Version::from_string(&version).map_err(SafetensorError::new_err)?;
                        if version >= Version::new(2, 1, 0) {
                            let dtype: PyObject = get_pydtype(torch, info.dtype, false)?;
                            tensor
                                .getattr(intern!(py, "untyped_storage"))?
                                .call0()?
                                .getattr(intern!(py, "byteswap"))?
                                .call1((dtype,))?;
                        } else if info.dtype == Dtype::BF16 {
                            return Err(SafetensorError::new_err(
                                "PyTorch 2.1 or later is required for big-endian machine and bfloat16 support.",
                            ));
                        } else {
                            let inplace_kwargs = [(intern!(py, "inplace"), false.into_py(py))]
                                .into_py_dict_bound(py);
                            let numpy = tensor
                                .getattr(intern!(py, "numpy"))?
                                .call0()?
                                .getattr("byteswap")?
                                .call((), Some(&inplace_kwargs))?;
                            tensor = torch.getattr(intern!(py, "from_numpy"))?.call1((numpy,))?;
                        }
                    }

                    tensor = tensor.getattr(intern!(py, "reshape"))?.call1((shape,))?;
                    if self.device != Device::Cpu {
                        let device: PyObject = self.device.clone().into_py(py);
                        let kwargs = PyDict::new_bound(py);
                        tensor = tensor.call_method("to", (device,), Some(&kwargs))?;
                    }
                    Ok(tensor.into_py(py))
                    // torch.asarray(storage[start + n : stop + n], dtype=torch.uint8).view(dtype=dtype).reshape(shape)
                })
            }
        }
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
        if let Some(&info) = self.metadata.tensors().get(name) {
            Ok(PySafeSlice {
                info: info.clone(),
                framework: self.framework.clone(),
                offset: self.offset,
                device: self.device.clone(),
                storage: self.storage.clone(),
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
    pub fn get_tensor(&self, name: &str) -> PyResult<PyObject> {
        self.inner()?.get_tensor(name)
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

    pub fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

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
    storage: Arc<Storage>,
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
        for item in &self.0 {
            write!(f, "{item}")?;
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
        let shape: PyObject = shape.into_py(py);
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
        let dtype = self.info.dtype;
        let dtype: PyObject = format!("{:?}", dtype).into_py(py);
        Ok(dtype)
    }

    pub fn __getitem__(&self, slices: &PyBound<'_, PyAny>) -> PyResult<PyObject> {
        match &self.storage.as_ref() {
            Storage::Mmap(mmap) => {
                let slices: Slice = slices.extract()?;
                let slices: Vec<SliceIndex> = match slices {
                    Slice::Slice(slice) => vec![slice],
                    Slice::Slices(slices) => slices,
                };
                let data = &mmap[self.info.data_offsets.0 + self.offset
                    ..self.info.data_offsets.1 + self.offset];

                let shape = self.info.shape.clone();

                let tensor = TensorView::new(self.info.dtype, self.info.shape.clone(), data)
                    .map_err(|e| {
                        SafetensorError::new_err(format!("Error preparing tensor view: {e:?}"))
                    })?;
                let slices: Vec<TensorIndexer> = slices
                    .into_iter()
                    .zip(shape)
                    .enumerate()
                    .map(slice_to_indexer)
                    .collect::<Result<_, _>>()?;

                let iterator = tensor.sliced_data(&slices).map_err(|e| {
                    SafetensorError::new_err(format!(
                        "Error during slicing {} with shape {:?}:  {:?}",
                        Disp(slices),
                        self.info.shape,
                        e
                    ))
                })?;
                let newshape = iterator.newshape();

                let mut offset = 0;
                let length = iterator.remaining_byte_len();
                Python::with_gil(|py| {
                    let array: PyObject =
                        PyByteArray::new_bound_with(py, length, |bytes: &mut [u8]| {
                            for slice in iterator {
                                let len = slice.len();
                                bytes[offset..offset + slice.len()].copy_from_slice(slice);
                                offset += len;
                            }
                            Ok(())
                        })?
                        .into_py(py);
                    create_tensor(
                        &self.framework,
                        self.info.dtype,
                        &newshape,
                        array,
                        &self.device,
                    )
                })
            }
            Storage::TorchStorage(storage) => Python::with_gil(|py| -> PyResult<PyObject> {
                let torch = get_module(py, &TORCH_MODULE)?;
                let dtype: PyObject = get_pydtype(torch, self.info.dtype, false)?;
                let torch_uint8: PyObject = get_pydtype(torch, Dtype::U8, false)?;
                let kwargs = [(intern!(py, "dtype"), torch_uint8)].into_py_dict_bound(py);
                let view_kwargs = [(intern!(py, "dtype"), dtype)].into_py_dict_bound(py);
                let shape = self.info.shape.to_vec();
                let shape: PyObject = shape.into_py(py);

                let start = (self.info.data_offsets.0 + self.offset) as isize;
                let stop = (self.info.data_offsets.1 + self.offset) as isize;
                let slice = PySlice::new_bound(py, start, stop, 1);
                let storage: &PyObject = storage
                    .get(py)
                    .ok_or_else(|| SafetensorError::new_err("Could not find storage"))?;
                let storage: &PyBound<'_, PyAny> = storage.bind(py);

                let storage_slice = storage
                    .getattr(intern!(py, "__getitem__"))?
                    .call1((slice,))?;

                let slices = slices.into_py(py);

                let sys = PyModule::import_bound(py, intern!(py, "sys"))?;
                let byteorder: String = sys.getattr(intern!(py, "byteorder"))?.extract()?;

                let mut tensor = torch
                    .getattr(intern!(py, "asarray"))?
                    .call((storage_slice,), Some(&kwargs))?
                    .getattr(intern!(py, "view"))?
                    .call((), Some(&view_kwargs))?;
                if byteorder == "big" {
                    let version: String = torch.getattr(intern!(py, "__version__"))?.extract()?;
                    let version =
                        Version::from_string(&version).map_err(SafetensorError::new_err)?;
                    if version >= Version::new(2, 1, 0) {
                        let dtype: PyObject = get_pydtype(torch, self.info.dtype, false)?;
                        tensor
                            .getattr(intern!(py, "untyped_storage"))?
                            .call0()?
                            .getattr(intern!(py, "byteswap"))?
                            .call1((dtype,))?;
                    } else if self.info.dtype == Dtype::BF16 {
                        return Err(SafetensorError::new_err(
                            "PyTorch 2.1 or later is required for big-endian machine and bfloat16 support.",
                        ));
                    } else {
                        let inplace_kwargs =
                            [(intern!(py, "inplace"), false.into_py(py))].into_py_dict_bound(py);
                        let numpy = tensor
                            .getattr(intern!(py, "numpy"))?
                            .call0()?
                            .getattr("byteswap")?
                            .call((), Some(&inplace_kwargs))?;
                        tensor = torch.getattr(intern!(py, "from_numpy"))?.call1((numpy,))?;
                    }
                }
                tensor = tensor
                    .getattr(intern!(py, "reshape"))?
                    .call1((shape,))?
                    .getattr(intern!(py, "__getitem__"))?
                    .call1((slices,))?;
                if self.device != Device::Cpu {
                    let device: PyObject = self.device.clone().into_py(py);
                    let kwargs = PyDict::new_bound(py);
                    tensor = tensor.call_method("to", (device,), Some(&kwargs))?;
                }
                Ok(tensor.into_py(py))
            }),
        }
    }
}

fn get_module<'a>(
    py: Python<'a>,
    cell: &'static GILOnceCell<Py<PyModule>>,
) -> PyResult<&'a PyBound<'a, PyModule>> {
    let module: &PyBound<'a, PyModule> = cell
        .get(py)
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
                    .get(py)
                    .ok_or_else(|| {
                        SafetensorError::new_err(format!("Could not find module {framework:?}",))
                    })?
                    .bind(py),
                false,
            ),
            _ => (
                NUMPY_MODULE
                    .get(py)
                    .ok_or_else(|| {
                        SafetensorError::new_err(format!("Could not find module {framework:?}",))
                    })?
                    .bind(py),
                true,
            ),
        };
        let dtype: PyObject = get_pydtype(module, dtype, is_numpy)?;
        let count: usize = shape.iter().product();
        let shape = shape.to_vec();
        let tensor = if count == 0 {
            // Torch==1.10 does not allow frombuffer on empty buffers so we create
            // the tensor manually.
            // let zeros = module.getattr(intern!(py, "zeros"))?;
            let shape: PyObject = shape.clone().into_py(py);
            let args = (shape,);
            let kwargs = [(intern!(py, "dtype"), dtype)].into_py_dict_bound(py);
            module.call_method("zeros", args, Some(&kwargs))?
        } else {
            // let frombuffer = module.getattr(intern!(py, "frombuffer"))?;
            let kwargs = [
                (intern!(py, "buffer"), array),
                (intern!(py, "dtype"), dtype),
            ]
            .into_py_dict_bound(py);
            module.call_method("frombuffer", (), Some(&kwargs))?
        };
        let mut tensor: PyBound<'_, PyAny> = tensor.call_method1("reshape", (shape,))?;
        let tensor = match framework {
            Framework::Flax => {
                let module = Python::with_gil(|py| -> PyResult<&Py<PyModule>> {
                    let module = PyModule::import_bound(py, intern!(py, "jax"))?;
                    Ok(FLAX_MODULE.get_or_init(py, || module.into()))
                })?
                .bind(py);
                module
                    .getattr(intern!(py, "numpy"))?
                    .getattr(intern!(py, "array"))?
                    .call1((tensor,))?
            }
            Framework::Tensorflow => {
                let module = Python::with_gil(|py| -> PyResult<&Py<PyModule>> {
                    let module = PyModule::import_bound(py, intern!(py, "tensorflow"))?;
                    Ok(TENSORFLOW_MODULE.get_or_init(py, || module.into()))
                })?
                .bind(py);
                module
                    .getattr(intern!(py, "convert_to_tensor"))?
                    .call1((tensor,))?
            }
            Framework::Mlx => {
                let module = Python::with_gil(|py| -> PyResult<&Py<PyModule>> {
                    let module = PyModule::import_bound(py, intern!(py, "mlx"))?;
                    Ok(MLX_MODULE.get_or_init(py, || module.into()))
                })?
                .bind(py);
                module
                    .getattr(intern!(py, "core"))?
                    // .getattr(intern!(py, "array"))?
                    .call_method1("array", (tensor,))?
            }
            Framework::Pytorch => {
                if device != &Device::Cpu {
                    let device: PyObject = device.clone().into_py(py);
                    let kwargs = PyDict::new_bound(py);
                    tensor = tensor.call_method("to", (device,), Some(&kwargs))?;
                }
                tensor
            }
            Framework::Numpy => tensor,
        };
        // let tensor = tensor.into_py_bound(py);
        Ok(tensor.into_py(py))
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
                    py.import_bound("builtins")?
                        .getattr(intern!(py, "bool"))?
                        .into()
                } else {
                    module.getattr(intern!(py, "bool"))?.into()
                }
            }
            Dtype::F8_E4M3 => module.getattr(intern!(py, "float8_e4m3fn"))?.into(),
            Dtype::F8_E5M2 => module.getattr(intern!(py, "float8_e5m2"))?.into(),
            dtype => {
                return Err(SafetensorError::new_err(format!(
                    "Dtype not understood: {dtype:?}"
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

/// A Python module implemented in Rust.
#[pymodule]
fn _safetensors_rust(m: &PyBound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(serialize, m)?)?;
    m.add_function(wrap_pyfunction!(serialize_file, m)?)?;
    m.add_function(wrap_pyfunction!(deserialize, m)?)?;
    m.add_class::<safe_open>()?;
    m.add(
        "SafetensorError",
        m.py().get_type_bound::<SafetensorError>(),
    )?;
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
