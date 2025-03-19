#[cfg(feature = "py311")]
mod py311 {
    use crate::SafetensorError;
    use pyo3::buffer::PyBuffer;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use pyo3::Bound as PyBound;
    use safetensors::tensor::Dtype;
    use safetensors::View;
    use std::borrow::Cow;
    pub(crate) struct PyView {
        shape: Vec<usize>,
        dtype: Dtype,
        data: PyBuffer<u8>,
        data_len: usize,
    }

    impl View for &PyView {
        fn data(&self) -> std::borrow::Cow<[u8]> {
            let slice = unsafe {
                std::slice::from_raw_parts(self.data.buf_ptr() as *mut u8, self.data.item_count())
            };
            Cow::Borrowed(slice)
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

    impl PyView {
        pub(crate) fn from(tensor_name: &str, tensor_desc: &PyBound<PyDict>) -> PyResult<Self> {
            let shape: Vec<usize> = tensor_desc
                .get_item("shape")?
                .ok_or_else(|| {
                    SafetensorError::new_err(format!("Missing `shape` in {tensor_desc:?}"))
                })?
                .extract()?;
            let pydata: PyBound<PyAny> = tensor_desc.get_item("data")?.ok_or_else(|| {
                SafetensorError::new_err(format!("Missing `data` in {tensor_desc:?}"))
            })?;
            // Make sure it's extractable first.
            let data: PyBuffer<u8> = pydata.extract()?;
            if !data.is_c_contiguous() {
                return Err(SafetensorError::new_err(format!(
                    "memoryview for {tensor_name} is not contiguous."
                )));
            }
            if !data.readonly() {
                return Err(SafetensorError::new_err(format!(
                    "memoryview for {tensor_name} is not readonly."
                )));
            }
            let data_len = data.item_count();
            // let data: PyBound<PyBytes> = pydata.extract()?;
            let pydtype = tensor_desc.get_item("dtype")?.ok_or_else(|| {
                SafetensorError::new_err(format!("Missing `dtype` in {tensor_desc:?}"))
            })?;
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
                dtype_str => {
                    return Err(SafetensorError::new_err(format!(
                        "dtype {dtype_str} is not covered",
                    )));
                }
            };

            Ok(Self {
                shape,
                dtype,
                data,
                data_len,
            })
        }
    }
}

#[cfg(feature = "py311")]
pub(crate) use py311::PyView;

#[cfg(feature = "py38")]
mod py38 {
    use crate::SafetensorError;
    use pyo3::prelude::*;
    use pyo3::types::{PyBytes, PyDict};
    use pyo3::Bound as PyBound;
    use safetensors::tensor::Dtype;
    use safetensors::View;
    use std::borrow::Cow;

    pub(crate) struct PyView<'a> {
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

    impl<'a> PyView<'a> {
        pub(crate) fn from(tensor_name: &str, tensor_desc: &PyBound<'a, PyDict>) -> PyResult<Self> {
            let shape: Vec<usize> = tensor_desc
                .get_item("shape")?
                .ok_or_else(|| {
                    SafetensorError::new_err(format!(
                        "Tensor {tensor_name}: missing `shape` in {tensor_desc:?}"
                    ))
                })?
                .extract()?;
            let pydata: PyBound<PyAny> = tensor_desc.get_item("data")?.ok_or_else(|| {
                SafetensorError::new_err(format!(
                    "Tensor {tensor_name}: missing `data` in {tensor_desc:?}"
                ))
            })?;
            // Make sure it's extractable first.
            let data: &[u8] = pydata.extract()?;
            let data_len = data.len();
            let data: PyBound<PyBytes> = pydata.extract()?;
            let pydtype = tensor_desc.get_item("dtype")?.ok_or_else(|| {
                SafetensorError::new_err(format!("Missing `dtype` in {tensor_desc:?}"))
            })?;
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
                dtype_str => {
                    return Err(SafetensorError::new_err(format!(
                        "dtype {dtype_str} is not covered",
                    )));
                }
            };

            Ok(Self {
                shape,
                dtype,
                data,
                data_len,
            })
        }
    }
}

#[cfg(feature = "py38")]
pub(crate) use py38::PyView;
