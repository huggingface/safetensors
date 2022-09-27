#![deny(missing_docs)]
//! Dummy doc
use memmap::{Mmap, MmapOptions};
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyList};
use safetensors::{Dtype, SafeTensors, Tensor};
use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;

fn prepare(tensor_dict: HashMap<String, &PyDict>) -> PyResult<HashMap<String, Tensor<'_>>> {
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

        let tensor = Tensor::new(data, dtype, shape);
        tensors.insert(tensor_name, tensor);
    }
    Ok(tensors)
}

#[pyfunction]
fn serialize<'a, 'b>(
    py: Python<'b>,
    tensor_dict: HashMap<String, &'a PyDict>,
) -> PyResult<&'b PyBytes> {
    let tensors = prepare(tensor_dict)?;
    let out = safetensors::serialize(&tensors);
    let pybytes = PyBytes::new(py, &out);
    Ok(pybytes)
}

#[pyfunction]
fn serialize_file(tensor_dict: HashMap<String, &PyDict>, filename: &str) -> PyResult<()> {
    let tensors = prepare(tensor_dict)?;
    safetensors::serialize_to_file(&tensors, filename)?;
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

#[pyfunction]
fn deserialize_file(
    py: Python,
    filename: &str,
) -> PyResult<Vec<(String, HashMap<String, PyObject>)>> {
    let file = File::open(filename)?;

    // SAFETY: Mmap is used to prevent allocating in Rust
    // before making a copy within Python.
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let deserialized = deserialize(py, &mmap);
    // Make sure mmap does not leak.
    drop(mmap);
    deserialized
}

use pyo3::types::PySlice;
use pyo3::{intern, PyErr};
use safetensors::slice::TensorIndexer;
use std::ops::Bound;

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

use pyo3::types::IntoPyDict;
use safetensors::{Metadata, TensorInfo, TensorView};

#[derive(Clone)]
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

    pub fn get_tensor(&self, py: Python, name: &str) -> PyResult<PyObject> {
        if let Some(info) = self.metadata.0.get(name) {
            let data =
                &self.mmap[info.data_offsets.0 + self.offset..info.data_offsets.1 + self.offset];

            let array: PyObject = PyByteArray::new(py, data).into_py(py);
            match self.framework {
                Framework::Pytorch => {
                    let module = PyModule::import(py, "torch")?;
                    let frombuffer = module.getattr("frombuffer")?;
                    let dtype: PyObject = match info.dtype {
                        Dtype::F32 => module.getattr("float32")?.into(),
                        _ => todo!("Pytorch dtypes"),
                    };
                    let kwargs = [("buffer", array), ("dtype", dtype)].into_py_dict(py);
                    let tensor = frombuffer.call((), Some(kwargs))?;
                    let shape = info.shape.clone();
                    let shape: PyObject = shape.into_py(py);
                    let tensor: PyObject = tensor.getattr("view")?.call1((shape,))?.into();
                    Ok(tensor)
                }
                _ => todo!(),
            }
        } else {
            Err(exceptions::PyException::new_err(format!(
                "File does not contain tensor {name}",
            )))
        }
    }

    pub fn get_slice(&self, name: &str) -> PyResult<PySafeSlice> {
        if let Some(info) = self.metadata.0.get(name) {
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

#[pymethods]
impl PySafeSlice {
    pub fn __getitem__(&self, py: Python, slices: Vec<&PySlice>) -> PyResult<PyObject> {
        let data = &self.mmap
            [self.info.data_offsets.0 + self.offset..self.info.data_offsets.1 + self.offset];
        let tensor = TensorView::new(&self.info.dtype, &self.info.shape, data);
        let slices: Vec<TensorIndexer> = slices
            .into_iter()
            .map(slice_to_indexer)
            .collect::<Result<_, _>>()?;
        let iterator = tensor.get_sliced_data(slices).map_err(|e| {
            exceptions::PyException::new_err(format!("Erro during slicing {:?}", e))
        })?;
        let newshape = iterator.newshape();

        let mut offset = 0;
        let array: PyObject =
            PyByteArray::new_with(py, iterator.remaining_byte_len(), |bytes: &mut [u8]| {
                for slice in iterator {
                    let len = slice.len();
                    bytes[offset..offset + slice.len()].copy_from_slice(slice);
                    offset += len
                }
                Ok(())
            })?
            .into_py(py);
        match self.framework {
            Framework::Pytorch => {
                let module = PyModule::import(py, "torch")?;
                let frombuffer = module.getattr("frombuffer")?;
                let dtype: PyObject = match self.info.dtype {
                    Dtype::F32 => module.getattr("float32")?.into(),
                    _ => todo!("Pytorch dtypes"),
                };
                let kwargs = [("buffer", array), ("dtype", dtype)].into_py_dict(py);
                let tensor = frombuffer.call((), Some(kwargs))?;
                let newshape: PyObject = newshape.into_py(py);
                let tensor: PyObject = tensor.getattr("view")?.call1((newshape,))?.into();
                Ok(tensor)
            }
            _ => todo!(),
        }
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn safetensors_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(serialize, m)?)?;
    m.add_function(wrap_pyfunction!(serialize_file, m)?)?;
    m.add_function(wrap_pyfunction!(deserialize, m)?)?;
    m.add_function(wrap_pyfunction!(deserialize_file, m)?)?;
    m.add_class::<safe_open>()?;
    Ok(())
}
