use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyList};
use safetensors::{Dtype, SafeTensor, SafeTensorBorrowed, Tensor};
use std::collections::HashMap;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn serialize<'a, 'b>(
    py: Python<'b>,
    tensor_dict: HashMap<String, &'a PyDict>,
) -> PyResult<&'b PyBytes> {
    let start = std::time::Instant::now();
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
                        "float32" => Dtype::F32,
                        "float64" => Dtype::F64,
                        "int32" => Dtype::I32,
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
    let out = SafeTensor::serialize(&tensors);
    let pybytes = PyBytes::new(py, &out);
    Ok(pybytes)
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn deserialize(py: Python, bytes: &[u8]) -> PyResult<Vec<(String, HashMap<String, PyObject>)>> {
    let start = std::time::Instant::now();
    let safetensor = SafeTensorBorrowed::deserialize(bytes).map_err(|e| {
        exceptions::PyException::new_err(format!("Error while deserializing: {:?}", e))
    })?;
    let mut items = vec![];

    for (tensor_name, tensor) in safetensor.tensors() {
        let mut map = HashMap::new();

        let pyshape: PyObject = PyList::new(py, tensor.shape.into_iter()).into();
        let pydtype: PyObject = format!("{:?}", tensor.dtype).into_py(py);

        let pydata: PyObject = PyByteArray::new(py, tensor.data).into();

        map.insert("shape".to_string(), pyshape);
        map.insert("dtype".to_string(), pydtype);
        map.insert("data".to_string(), pydata);
        items.push((tensor_name, map));
    }
    Ok(items)
}

/// A Python module implemented in Rust.
#[pymodule]
fn safetensors_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(serialize, m)?)?;
    m.add_function(wrap_pyfunction!(deserialize, m)?)?;
    Ok(())
}
