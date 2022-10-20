#![deny(missing_docs)]
//! Dummy doc
use memmap::{Mmap, MmapOptions};
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use pyo3::types::PySlice;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyList};
use pyo3::{intern, PyErr};
use safetensors::slice::TensorIndexer;
use safetensors::tensor::{Dtype, Metadata, SafeTensors, TensorInfo, TensorView};
use std::collections::HashMap;
use std::fs::File;
use std::ops::Bound;
use std::sync::Arc;

fn prepare(tensor_dict: HashMap<String, &PyDict>) -> PyResult<HashMap<String, TensorView<'_>>> {
    let mut tensors = HashMap::new();
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

        let tensor = TensorView::new(dtype, shape, data).map_err(|e| {
            exceptions::PyException::new_err(format!(
                "Error while processing tensor {tensor_name:?} {:?}",
                e
            ))
        })?;
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
    let out = safetensors::tensor::serialize(&tensors, &metadata);
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
    safetensors::tensor::serialize_to_file(&tensors, &metadata, filename)?;
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

#[derive(Debug, Clone)]
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

#[pyclass]
#[allow(non_camel_case_types)]
struct safe_open {
    metadata: Metadata,
    offset: usize,
    framework: Framework,
    mmap: Arc<Mmap>,
}

#[pymethods]
impl safe_open {
    #[new]
    fn new(filename: &str, framework: Framework) -> PyResult<Self> {
        let file = File::open(filename)?;

        // SAFETY: Mmap is used to prevent allocating in Rust
        // before making a copy within Python.
        let buffer = unsafe { MmapOptions::new().map(&file)? };

        let arr: [u8; 8] = [
            buffer[0], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5], buffer[6], buffer[7],
        ];
        let n = u64::from_le_bytes(arr) as usize;
        let string = std::str::from_utf8(&buffer[8..8 + n]).map_err(|e| {
            exceptions::PyException::new_err(format!("Error while deserializing header: {:?}", e))
        })?;
        let metadata: Metadata = serde_json::from_str(string).map_err(|e| {
            exceptions::PyException::new_err(format!("Error while deserializing metadata: {:?}", e))
        })?;

        let offset = n + 8;

        Ok(Self {
            metadata,
            offset,
            framework,
            mmap: Arc::new(buffer),
        })
    }

    pub fn metadata(&self) -> Option<HashMap<String, String>> {
        self.metadata.metadata().clone()
    }

    pub fn keys(&self) -> PyResult<Vec<String>> {
        let mut keys: Vec<_> = self.metadata.tensors().keys().cloned().collect();
        keys.sort();
        Ok(keys)
    }

    pub fn get_tensor(&self, py: Python, name: &str) -> PyResult<PyObject> {
        if let Some(info) = self.metadata.tensors().get(name) {
            let data =
                &self.mmap[info.data_offsets.0 + self.offset..info.data_offsets.1 + self.offset];

            let tensor = TensorView::new(info.dtype, info.shape.clone(), data).map_err(|e| {
                exceptions::PyException::new_err(format!("Error when creating TensorView {:?}", e))
            })?;
            let array: PyObject = PyByteArray::new(py, tensor.get_data()).into_py(py);

            create_tensor(py, &self.framework, info.dtype, &info.shape, array)
        } else {
            Err(exceptions::PyException::new_err(format!(
                "File does not contain tensor {name}",
            )))
        }
    }

    pub fn get_tensor_info(&self, py: Python, name: &str) -> PyResult<PyObject> {
        if let Some(info) = self.metadata.tensors().get(name) {
            let shape = info.shape.clone();
            let dtype = info.dtype.to_string();
            let data_offsets = vec![
                info.data_offsets.0 + self.offset,
                info.data_offsets.1 + self.offset,
            ];
            let py_data_offsets: PyObject = PyList::new(py, data_offsets).into();
            let py_shape: PyObject = PyList::new(py, shape).into();
            let py_dtype: PyObject = dtype.to_object(py);

            let dict: PyObject = [
                ("data_offsets", py_data_offsets),
                ("shape", py_shape),
                ("dtype", py_dtype),
            ]
            .into_py_dict(py)
            .into();

            Ok(dict)
        } else {
            Err(exceptions::PyException::new_err(format!(
                "File does not contain tensor {name}",
            )))
        }
    }

    pub fn get_slice(&self, name: &str) -> PyResult<PySafeSlice> {
        if let Some(info) = self.metadata.tensors().get(name) {
            Ok(PySafeSlice {
                info: info.clone(),
                framework: self.framework.clone(),
                offset: self.offset,
                mmap: self.mmap.clone(),
            })
        } else {
            Err(exceptions::PyException::new_err(format!(
                "File does not contain tensor {name}",
            )))
        }
    }

    pub fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    pub fn __exit__(&mut self, _exc_type: PyObject, _exc_value: PyObject, _traceback: PyObject) {}
}

#[pyclass]
struct PySafeSlice {
    info: TensorInfo,
    framework: Framework,
    offset: usize,
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
    pub fn __getitem__(&self, py: Python, slices: Slice) -> PyResult<PyObject> {
        let slices: Vec<&PySlice> = match slices {
            Slice::Slice(slice) => vec![slice],
            Slice::Slices(slices) => slices,
        };
        let data = &self.mmap
            [self.info.data_offsets.0 + self.offset..self.info.data_offsets.1 + self.offset];

        let tensor =
            TensorView::new(self.info.dtype, self.info.shape.clone(), data).map_err(|e| {
                exceptions::PyException::new_err(format!("Error when creating TensorView {:?}", e))
            })?;
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
        let array: PyObject = PyByteArray::new_with(py, length, |bytes: &mut [u8]| {
            for slice in iterator {
                let len = slice.len();
                bytes[offset..offset + slice.len()].copy_from_slice(slice);
                offset += len;
            }
            Ok(())
        })?
        .into_py(py);
        create_tensor(py, &self.framework, self.info.dtype, &newshape, array)
    }
}

fn create_tensor(
    py: Python,
    framework: &Framework,
    dtype: Dtype,
    shape: &[usize],
    array: PyObject,
) -> PyResult<PyObject> {
    match framework {
        Framework::Pytorch | Framework::Numpy => {
            let module_name = match framework {
                Framework::Numpy => "numpy",
                Framework::Pytorch => "torch",
                _ => unreachable!(),
            };
            let module = PyModule::import(py, module_name)?;
            let frombuffer = module.getattr("frombuffer")?;
            let dtype: PyObject = get_pydtype(module, dtype)?;
            let kwargs = [("buffer", array), ("dtype", dtype)].into_py_dict(py);
            let tensor = frombuffer.call((), Some(kwargs))?;
            let shape = shape.to_vec();
            let shape: PyObject = shape.into_py(py);
            let tensor: PyObject = tensor.getattr("reshape")?.call1((shape,))?.into();
            Ok(tensor)
        }
        framework => todo!("{framework:?}"),
    }
}

fn get_pydtype(module: &PyModule, dtype: Dtype) -> PyResult<PyObject> {
    let dtype: PyObject = match dtype {
        Dtype::F64 => module.getattr("float64")?.into(),
        Dtype::F32 => module.getattr("float32")?.into(),
        Dtype::BF16 => module.getattr("bfloat16")?.into(),
        Dtype::F16 => module.getattr("float16")?.into(),
        Dtype::U64 => module.getattr("uint64")?.into(),
        Dtype::I64 => module.getattr("int64")?.into(),
        Dtype::U32 => module.getattr("uint32")?.into(),
        Dtype::I32 => module.getattr("int32")?.into(),
        Dtype::U16 => module.getattr("uint16")?.into(),
        Dtype::I16 => module.getattr("int16")?.into(),
        Dtype::U8 => module.getattr("uint8")?.into(),
        Dtype::I8 => module.getattr("int8")?.into(),
        Dtype::BOOL => module.getattr("bool")?.into(),
        dtype => todo!("Dtype {dtype:?}"),
    };
    Ok(dtype)
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
