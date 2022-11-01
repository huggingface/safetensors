#![deny(missing_docs)]
//! Dummy doc
use libloading::{Library, Symbol};
use memmap::{Mmap, MmapOptions};
use pyo3::exceptions;
use pyo3::once_cell::GILOnceCell;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use pyo3::types::PySlice;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyList};
use pyo3::{intern, PyErr};
use safetensors::slice::TensorIndexer;
use safetensors::tensor::{Dtype, Metadata, SafeTensors, TensorInfo, TensorView};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::iter::FromIterator;
use std::ops::Bound;
use std::sync::Arc;

#[repr(C)]
enum cudaMemcpyKind {
    _HostToHost = 0,
    HostToDevice = 1,
    _DeviceToHost = 2,
    _Default = 3,
}

type MemcpyFn = unsafe extern "C" fn(
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    count: usize,
    kind: cudaMemcpyKind,
) -> std::ffi::c_uint;
static TORCH_MODULE: GILOnceCell<Py<PyModule>> = GILOnceCell::new();
static NUMPY_MODULE: GILOnceCell<Py<PyModule>> = GILOnceCell::new();
static CUDA_MEMCPY: GILOnceCell<Option<Symbol<MemcpyFn>>> = GILOnceCell::new();

fn prepare(tensor_dict: HashMap<String, &PyDict>) -> PyResult<BTreeMap<String, TensorView<'_>>> {
    let mut tensors = BTreeMap::new();
    for (tensor_name, tensor_desc) in tensor_dict {
        let mut shape: Vec<usize> = vec![];
        let mut dtype = Dtype::F32;
        let mut data: &[u8] = &[];
        for (key, value) in tensor_desc {
            let key: &str = key.extract()?;
            match key {
                "shape" => shape = value.extract()?,
                "dtype" => {
                    let value: &str = value.extract()?;
                    dtype = match value {
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
                        dtype_str => {
                            unimplemented!("Did not cover this dtype: {}", dtype_str)
                        }
                    }
                }
                "data" => data = value.extract()?,
                _ => println!("Ignored unknown kwarg option {}", key),
            };
        }
        let tensor = TensorView::new(dtype, shape, data);
        tensors.insert(tensor_name, tensor);
    }
    Ok(tensors)
}

#[pyfunction]
fn serialize<'a, 'b>(
    py: Python<'b>,
    tensor_dict: HashMap<String, &'a PyDict>,
    metadata: Option<HashMap<String, String>>,
) -> PyResult<&'b PyBytes> {
    let tensors = prepare(tensor_dict)?;
    let metadata_btreemap = metadata.map(|data| BTreeMap::from_iter(data.into_iter()));
    let out = safetensors::tensor::serialize(&tensors, &metadata_btreemap).map_err(|e| {
        exceptions::PyException::new_err(format!("Error while serializing: {:?}", e))
    })?;
    let pybytes = PyBytes::new(py, &out);
    Ok(pybytes)
}

#[pyfunction]
fn serialize_file(
    tensor_dict: HashMap<String, &PyDict>,
    filename: &str,
    metadata: Option<HashMap<String, String>>,
) -> PyResult<()> {
    let tensors = prepare(tensor_dict)?;
    let metadata_btreemap = metadata.map(|data| BTreeMap::from_iter(data.into_iter()));
    safetensors::tensor::serialize_to_file(&tensors, &metadata_btreemap, filename).map_err(
        |e| exceptions::PyException::new_err(format!("Error while serializing: {:?}", e)),
    )?;
    Ok(())
}

#[pyfunction]
fn deserialize(py: Python, bytes: &[u8]) -> PyResult<Vec<(String, HashMap<String, PyObject>)>> {
    let safetensor = SafeTensors::deserialize(bytes).map_err(|e| {
        exceptions::PyException::new_err(format!("Error while deserializing: {:?}", e))
    })?;
    let mut items = vec![];

    for (tensor_name, tensor) in safetensor.tensors() {
        let mut map = HashMap::new();

        let pyshape: PyObject = PyList::new(py, tensor.get_shape().iter()).into();
        let pydtype: PyObject = format!("{:?}", tensor.get_dtype()).into_py(py);

        let pydata: PyObject = PyByteArray::new(py, tensor.get_data()).into();

        map.insert("shape".to_string(), pyshape);
        map.insert("dtype".to_string(), pydtype);
        map.insert("data".to_string(), pydata);
        items.push((tensor_name, map));
    }
    Ok(items)
}

fn slice_to_indexer(slice: &PySlice) -> Result<TensorIndexer, PyErr> {
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

#[derive(Debug, Clone, PartialEq, Eq)]
enum Framework {
    Pytorch,
    Numpy,
    Tensorflow,
    Jax,
}

impl<'source> FromPyObject<'source> for Framework {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let name: String = ob.extract()?;
        match &name[..] {
            "pt" => Ok(Framework::Pytorch),
            "np" => Ok(Framework::Numpy),
            "tf" => Ok(Framework::Tensorflow),
            "jax" => Ok(Framework::Jax),
            name => Err(exceptions::PyException::new_err(format!(
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
}

impl<'source> FromPyObject<'source> for Device {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        if let Ok(name) = ob.extract::<String>() {
            match &name[..] {
                "cpu" => Ok(Device::Cpu),
                "cuda" => Ok(Device::Cuda(0)),
                "mps" => Ok(Device::Mps),
                name if name.starts_with("cuda:") => {
                    let tokens: Vec<_> = name.split(':').collect();
                    if tokens.len() == 2 {
                        let device: usize = tokens[1].parse()?;
                        Ok(Device::Cuda(device))
                    } else {
                        Err(exceptions::PyException::new_err(format!(
                            "device {name} is invalid"
                        )))
                    }
                }
                name => Err(exceptions::PyException::new_err(format!(
                    "device {name} is invalid"
                ))),
            }
        } else if let Ok(number) = ob.extract::<usize>() {
            Ok(Device::Cuda(number))
        } else {
            Err(exceptions::PyException::new_err(format!(
                "device {ob} is invalid"
            )))
        }
    }
}

impl IntoPy<PyObject> for Device {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            Device::Cpu => "cpu".into_py(py),
            Device::Cuda(n) => format!("cuda:{n}").into_py(py),
            Device::Mps => "mps".into_py(py),
        }
    }
}

#[pyclass]
#[allow(non_camel_case_types)]
struct safe_open {
    metadata: Metadata,
    offset: usize,
    framework: Framework,
    device: Device,
    mmap: Arc<Mmap>,
}

fn create_empty_tensor_pt<'a>(
    module: &'a PyModule,
    shape: &[usize],
    dtype: Dtype,
    device: &Device,
) -> PyResult<&'a PyAny> {
    let py = module.py();
    let shape = shape.to_vec();
    let empty = module.getattr(intern!(py, "empty"))?;
    let dtype: PyObject = get_pydtype(module, dtype)?;
    let shape: PyObject = shape.into_py(py);
    let device: PyObject = device.clone().into_py(py);
    let kwargs = [
        (intern!(py, "dtype"), dtype),
        (intern!(py, "device"), device),
    ]
    .into_py_dict(py);
    let tensor = empty.call((shape,), Some(kwargs))?;

    Ok(tensor)
}

fn find_cudart(module: &PyModule) -> Option<Library> {
    let var = std::env::var("SAFETENSORS_FAST_GPU").ok()?;
    if var != "1" {
        return None;
    }
    let path: std::path::PathBuf = module
        .getattr(intern!(module.py(), "_C"))
        .ok()?
        .getattr(intern!(module.py(), "__file__"))
        .ok()?
        .extract()
        .ok()?;
    // SAFETY: This is unsafe because the library might run arbitrary code
    // So it's really important to make sure we are targeting the correct
    // library.
    let lib = unsafe { Library::new(path).ok()? };
    Some(lib)
}

fn find_cuda_memcpy<'py>(module: &PyModule) -> Option<Symbol<'py, MemcpyFn>> {
    let cudart = find_cudart(module)?;
    // Leaking the library so that the reference becomes static
    let cudart = Box::leak(Box::new(cudart));
    // SAFETY: This is unsafe because the library might run arbitrary code
    // So it's really important to make sure we are targeting the correct
    // library.
    let cuda_memcpy = unsafe { cudart.get(b"cudaMemcpy").ok() };
    cuda_memcpy
}

fn create_cuda_unsafe_tensor(
    module: &PyModule,
    cuda_memcpy: &Symbol<MemcpyFn>,
    info: &TensorInfo,
    device: &Device,
    data: &[u8],
) -> PyResult<PyObject> {
    let tensor = create_empty_tensor_pt(module, &info.shape, info.dtype, device)?;

    let data_ptr_fn = tensor.getattr("data_ptr")?;
    let data_ptr: usize = data_ptr_fn.call0()?.extract()?;
    check_cuda(cuda_memcpy, data_ptr, data, module);
    let tensor: PyObject = tensor.into_py(module.py());
    Ok(tensor)
}

fn check_cuda(cuda_memcpy: &Symbol<MemcpyFn>, dst: usize, src: &[u8], module: &PyModule) {
    let dst = dst as *mut std::ffi::c_void;
    let count = src.len();
    let src_ptr = src.as_ptr() as *const std::ffi::c_void;
    // SAFETY: Here we have a correct library, we successfully called memcpy,
    // but somehow the call failed. This is really worrying since Pytorch is
    // responsible for allocating the memory.
    let out = unsafe { cuda_memcpy(dst, src_ptr, count, cudaMemcpyKind::HostToDevice) };
    if out != 0 {
        let string = match get_error_string(module, out) {
            Ok(string) => string,
            Err(_) => format!("{}", out),
        };
        println!(
                "We tried to set your tensor fast, but there was a cuda error, This could have corrupted your GPU ram, aborting to prevent further errors {string:?}"
            );
        std::process::abort();
    }
}

fn create_cuda_unsafe_tensor_from_slice(
    module: &PyModule,
    cuda_memcpy: &Symbol<MemcpyFn>,
    shape: &[usize],
    dtype: Dtype,
    device: &Device,
    iterator: safetensors::slice::SliceIterator,
) -> PyResult<PyObject> {
    let tensor = create_empty_tensor_pt(module, shape, dtype, device)?;

    let mut offset = 0;
    let data_ptr_fn = tensor.getattr("data_ptr")?;
    let data_ptr: usize = data_ptr_fn.call0()?.extract()?;
    for slice in iterator {
        let len = slice.len();
        check_cuda(cuda_memcpy, data_ptr + offset, slice, module);
        offset += len;
    }
    let tensor: PyObject = tensor.into_py(module.py());
    Ok(tensor)
}

fn get_error_string(module: &PyModule, out: u32) -> PyResult<String> {
    module
        .getattr(intern!(module.py(), "cuda"))?
        .getattr(intern!(module.py(), "CudaError"))?
        .call1((out.into_py(module.py()),))?
        .getattr(intern!(module.py(), "__str__"))?
        .call0()?
        .extract()
}

#[pymethods]
impl safe_open {
    #[new]
    fn new(filename: &str, framework: Framework, device: Option<Device>) -> PyResult<Self> {
        let file = File::open(filename)?;
        let device = device.unwrap_or(Device::Cpu);

        if device != Device::Cpu && framework != Framework::Pytorch {
            return Err(exceptions::PyException::new_err(format!(
                "Device {device:?} is not support for framework {framework:?}",
            )));
        }

        // SAFETY: Mmap is used to prevent allocating in Rust
        // before making a copy within Python.
        let buffer = unsafe { MmapOptions::new().map(&file)? };

        let (n, metadata) = SafeTensors::read_metadata(&buffer).map_err(|e| {
            exceptions::PyException::new_err(format!("Error while deserializing header: {:?}", e))
        })?;

        let offset = n + 8;

        Python::with_gil(|py| -> PyResult<()> {
            match framework {
                Framework::Pytorch => {
                    let module = PyModule::import(py, intern!(py, "torch"))?;
                    TORCH_MODULE.get_or_init(py, || module.into())
                }
                _ => {
                    let module = PyModule::import(py, intern!(py, "numpy"))?;
                    NUMPY_MODULE.get_or_init(py, || module.into())
                }
            };

            if let (Device::Cuda(_), Framework::Pytorch) = (&device, &framework) {
                let module: &PyModule = get_module(py, &TORCH_MODULE)?;
                CUDA_MEMCPY.get_or_init(py, || find_cuda_memcpy(module));
            }
            Ok(())
        })?;

        Ok(Self {
            metadata,
            offset,
            framework,
            device,
            mmap: Arc::new(buffer),
        })
    }

    pub fn metadata(&self) -> Option<BTreeMap<String, String>> {
        self.metadata.metadata().clone()
    }

    pub fn keys(&self) -> PyResult<Vec<String>> {
        let mut keys: Vec<_> = self.metadata.tensors().keys().cloned().collect();
        keys.sort();
        Ok(keys)
    }

    pub fn get_tensor(&self, name: &str) -> PyResult<PyObject> {
        let info = self.metadata.tensors().get(name).ok_or_else(|| {
            exceptions::PyException::new_err(format!("File does not contain tensor {name}",))
        })?;

        let data = &self.mmap[info.data_offsets.0 + self.offset..info.data_offsets.1 + self.offset];

        Python::with_gil(|py| -> PyResult<PyObject> {
            match (&self.device, &self.framework, CUDA_MEMCPY.get(py)) {
                (Device::Cuda(_), Framework::Pytorch, Some(Some(cuda_memcpy))) => {
                    let module = get_module(py, &TORCH_MODULE)?;
                    create_cuda_unsafe_tensor(module, cuda_memcpy, info, &self.device, data)
                }
                _ => {
                    let array: PyObject =
                        Python::with_gil(|py| PyByteArray::new(py, data).into_py(py));

                    create_tensor(
                        &self.framework,
                        info.dtype,
                        &info.shape,
                        array,
                        &self.device,
                    )
                }
            }
        })
    }

    pub fn get_slice(&self, name: &str) -> PyResult<PySafeSlice> {
        if let Some(info) = self.metadata.tensors().get(name) {
            Ok(PySafeSlice {
                info: info.clone(),
                framework: self.framework.clone(),
                offset: self.offset,
                device: self.device.clone(),
                mmap: self.mmap.clone(),
            })
        } else {
            Err(exceptions::PyException::new_err(format!(
                "File does not contain tensor {name}",
            )))
        }
    }

    pub fn __enter__(slf: Py<Self>) -> Py<Self> {
        // SAFETY: This code is extremely important to the GPU fast load.
        // Cuda uses a context to select the device you are writing on.
        // PyTorch uses this function to create and use said context.
        // Without this, we instantiate the empty buffer on the correct GPU
        // But we fail to override the proper memory location since the context
        // is removed by python.
        // Using this sets the Cuda context once and for all for the entirety
        // of the context manager lifecycle.
        Python::with_gil(|py| -> PyResult<()> {
            let _self: &safe_open = &slf.borrow(py);
            if let (Device::Cuda(_), Framework::Pytorch) = (&_self.device, &_self.framework) {
                let module = get_module(py, &TORCH_MODULE)?;
                let device: PyObject = _self.device.clone().into_py(py);
                let torch_device = module
                    .getattr(intern!(py, "cuda"))?
                    .getattr(intern!(py, "device"))?;
                let lock = torch_device.call1((device,))?;
                lock.call_method0(intern!(py, "__enter__"))?;
            }
            Ok(())
        })
        .ok();
        slf
    }

    pub fn __exit__(&mut self, _exc_type: PyObject, _exc_value: PyObject, _traceback: PyObject) {
        if let (Device::Cuda(_), Framework::Pytorch) = (&self.device, &self.framework) {
            Python::with_gil(|py| -> PyResult<()> {
                let module = get_module(py, &TORCH_MODULE)?;
                let device: PyObject = self.device.clone().into_py(py);
                let torch_device = module
                    .getattr(intern!(py, "cuda"))?
                    .getattr(intern!(py, "device"))?;
                let none = py.None();
                let lock = torch_device.call1((device,))?;
                lock.call_method1(intern!(py, "__exit__"), (&none, &none, &none))?;
                Ok(())
            })
            .ok();
        }
    }
}

#[pyclass]
struct PySafeSlice {
    info: TensorInfo,
    framework: Framework,
    offset: usize,
    device: Device,
    mmap: Arc<Mmap>,
}

#[derive(FromPyObject)]
enum Slice<'a> {
    // Index(usize),
    Slice(&'a PySlice),
    Slices(Vec<&'a PySlice>),
}

#[pymethods]
impl PySafeSlice {
    pub fn get_shape(&self, py: Python) -> PyResult<PyObject> {
        let shape = self.info.shape.clone();
        let shape: PyObject = shape.into_py(py);
        Ok(shape)
    }

    pub fn __getitem__(&self, slices: Slice) -> PyResult<PyObject> {
        let slices: Vec<&PySlice> = match slices {
            Slice::Slice(slice) => vec![slice],
            Slice::Slices(slices) => slices,
        };
        let data = &self.mmap
            [self.info.data_offsets.0 + self.offset..self.info.data_offsets.1 + self.offset];

        let tensor = TensorView::new(self.info.dtype, self.info.shape.clone(), data);
        let slices: Vec<TensorIndexer> = slices
            .into_iter()
            .map(slice_to_indexer)
            .collect::<Result<_, _>>()?;

        let iterator = tensor.get_sliced_data(slices.clone()).map_err(|e| {
            exceptions::PyException::new_err(format!(
                "Error during slicing {slices:?} vs {:?}:  {:?}",
                self.info.shape, e
            ))
        })?;
        let newshape = iterator.newshape();

        let mut offset = 0;
        let length = iterator.remaining_byte_len();

        Python::with_gil(
            |py| match (&self.device, &self.framework, CUDA_MEMCPY.get(py)) {
                (Device::Cuda(_), Framework::Pytorch, Some(Some(cuda_memcpy))) => {
                    let module = get_module(py, &TORCH_MODULE)?;
                    create_cuda_unsafe_tensor_from_slice(
                        module,
                        cuda_memcpy,
                        &newshape,
                        self.info.dtype,
                        &self.device,
                        iterator,
                    )
                }
                _ => {
                    let array: PyObject =
                        PyByteArray::new_with(py, length, |bytes: &mut [u8]| {
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
                }
            },
        )
    }
}

fn get_module<'a>(
    py: Python<'a>,
    cell: &'static GILOnceCell<Py<PyModule>>,
) -> PyResult<&'a PyModule> {
    let module: &PyModule = cell
        .get(py)
        .ok_or_else(|| exceptions::PyException::new_err("Could not find module"))?
        .as_ref(py);
    Ok(module)
}

fn create_tensor(
    framework: &Framework,
    dtype: Dtype,
    shape: &[usize],
    array: PyObject,
    device: &Device,
) -> PyResult<PyObject> {
    match framework {
        Framework::Pytorch | Framework::Numpy => Python::with_gil(|py| -> PyResult<PyObject> {
            let module: &PyModule = match framework {
                Framework::Numpy => NUMPY_MODULE.get(py),
                Framework::Pytorch => TORCH_MODULE.get(py),
                _ => unreachable!(),
            }
            .ok_or_else(|| {
                exceptions::PyException::new_err(format!("Could not find module {framework:?}",))
            })?
            .as_ref(py);
            let frombuffer = module.getattr(intern!(py, "frombuffer"))?;
            let dtype: PyObject = get_pydtype(module, dtype)?;
            let kwargs = [
                (intern!(py, "buffer"), array),
                (intern!(py, "dtype"), dtype),
            ]
            .into_py_dict(py);
            let tensor = frombuffer.call((), Some(kwargs))?;
            let shape = shape.to_vec();
            let shape: PyObject = shape.into_py(py);
            let mut tensor: &PyAny = tensor.getattr(intern!(py, "reshape"))?.call1((shape,))?;
            if device != &Device::Cpu {
                let device: PyObject = device.clone().into_py(py);
                let kwargs = PyDict::new(py);
                tensor = tensor
                    .getattr(intern!(py, "to"))?
                    .call((device,), Some(kwargs))?;
            }
            let tensor = tensor.into_py(py);
            Ok(tensor)
        }),
        framework => todo!("{framework:?}"),
    }
}

fn get_pydtype(module: &PyModule, dtype: Dtype) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let dtype: PyObject = match dtype {
            Dtype::F64 => module.getattr(intern!(py, "float64"))?.into(),
            Dtype::F32 => module.getattr(intern!(py, "float32"))?.into(),
            Dtype::BF16 => module.getattr(intern!(py, "bfloat16"))?.into(),
            Dtype::F16 => module.getattr(intern!(py, "float16"))?.into(),
            Dtype::U64 => module.getattr(intern!(py, "uint64"))?.into(),
            Dtype::I64 => module.getattr(intern!(py, "int64"))?.into(),
            Dtype::U32 => module.getattr(intern!(py, "uint32"))?.into(),
            Dtype::I32 => module.getattr(intern!(py, "int32"))?.into(),
            Dtype::U16 => module.getattr(intern!(py, "uint16"))?.into(),
            Dtype::I16 => module.getattr(intern!(py, "int16"))?.into(),
            Dtype::U8 => module.getattr(intern!(py, "uint8"))?.into(),
            Dtype::I8 => module.getattr(intern!(py, "int8"))?.into(),
            Dtype::BOOL => module.getattr(intern!(py, "bool"))?.into(),
            dtype => todo!("Dtype {dtype:?}"),
        };
        Ok(dtype)
    })
}
/// A Python module implemented in Rust.
#[pymodule]
fn safetensors_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(serialize, m)?)?;
    m.add_function(wrap_pyfunction!(serialize_file, m)?)?;
    m.add_function(wrap_pyfunction!(deserialize, m)?)?;
    m.add_class::<safe_open>()?;
    Ok(())
}
