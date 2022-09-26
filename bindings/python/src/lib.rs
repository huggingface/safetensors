#![deny(missing_docs)]
//! Dummy doc
use memmap::MmapOptions;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyList};
use safetensors::{Dtype, SafeTensors, Tensor};
use std::collections::HashMap;
use std::fs::File;

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

/// A Python module implemented in Rust.
#[pymodule]
fn safetensors_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(serialize, m)?)?;
    m.add_function(wrap_pyfunction!(serialize_file, m)?)?;
    m.add_function(wrap_pyfunction!(deserialize, m)?)?;
    m.add_function(wrap_pyfunction!(deserialize_file, m)?)?;
    Ok(())
}
