use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug)]
pub enum SafeTensorError {
    InvalidHeader,
}

pub struct SafeTensor {
    metadata: Metadata,
    offset: usize,
    data: Vec<u8>,
}

impl SafeTensor {
    pub fn serialize(data: &HashMap<String, Tensor>) -> Vec<u8> {
        let mut tensors: Vec<&Tensor> = vec![];
        let mut hmetadata = HashMap::new();
        let mut offset = 0;
        for (name, tensor) in data {
            let n = tensor.data.len();

            let tensor_info = TensorInfo {
                dtype: tensor.dtype.clone(),
                shape: tensor.shape.clone(),
                data_offsets: (offset, offset + n),
            };
            offset += n;
            hmetadata.insert(name.to_string(), tensor_info);
            tensors.push(tensor);
        }

        let metadata: Metadata = Metadata(hmetadata);

        let metadata_buf = serde_json::to_string(&metadata).unwrap().into_bytes();

        let mut buffer: Vec<u8> = Vec::with_capacity(1 + metadata_buf.len() + offset);
        let n: u64 = metadata_buf.len() as u64;
        buffer.extend(&n.to_le_bytes().to_vec());
        buffer.extend(&metadata_buf);
        for tensor in tensors {
            buffer.extend(tensor.data);
        }
        buffer
    }

    pub fn deserialize(buffer: Vec<u8>) -> Result<SafeTensor, SafeTensorError> {
        let arr: [u8; 8] = [
            buffer[0], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5], buffer[6], buffer[7],
        ];
        let n = u64::from_le_bytes(arr) as usize;
        let string =
            std::str::from_utf8(&buffer[8..8 + n]).map_err(|_| SafeTensorError::InvalidHeader)?;
        let metadata: Metadata =
            serde_json::from_str(string).map_err(|_| SafeTensorError::InvalidHeader)?;
        Ok(SafeTensor {
            metadata,
            offset: n + 8,
            data: buffer,
        })
    }

    pub fn tensors<'data>(&'data self) -> Vec<(String, TensorView<'data>)> {
        let mut tensors = vec![];
        for (name, info) in &self.metadata.0 {
            let tensorview = TensorView {
                dtype: &info.dtype,
                shape: &info.shape,
                data: &self.data
                    [info.data_offsets.0 + self.offset..info.data_offsets.1 + self.offset],
            };
            tensors.push((name.to_string(), tensorview));
        }
        tensors
    }
}

pub struct SafeTensorBorrowed<'data> {
    metadata: Metadata,
    offset: usize,
    data: &'data [u8],
}
impl<'data> SafeTensorBorrowed<'data> {
    pub fn deserialize<'in_data>(buffer: &'in_data [u8]) -> Result<Self, SafeTensorError>
    where
        'in_data: 'data,
    {
        let arr: [u8; 8] = [
            buffer[0], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5], buffer[6], buffer[7],
        ];
        let n = u64::from_le_bytes(arr) as usize;
        let string =
            std::str::from_utf8(&buffer[8..8 + n]).map_err(|_| SafeTensorError::InvalidHeader)?;
        let metadata: Metadata =
            serde_json::from_str(string).map_err(|_| SafeTensorError::InvalidHeader)?;
        Ok(Self {
            metadata,
            offset: n + 8,
            data: buffer,
        })
    }

    pub fn tensors(&self) -> Vec<(String, TensorView<'_>)> {
        let mut tensors = vec![];
        for (name, info) in &self.metadata.0 {
            let tensorview = TensorView {
                dtype: &info.dtype,
                shape: &info.shape,
                data: &self.data
                    [info.data_offsets.0 + self.offset..info.data_offsets.1 + self.offset],
            };
            tensors.push((name.to_string(), tensorview));
        }
        tensors
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct Metadata(HashMap<String, TensorInfo>);

#[derive(Debug)]
pub struct TensorView<'data> {
    pub dtype: &'data Dtype,
    pub shape: &'data [usize],
    pub data: &'data [u8],
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct TensorInfo {
    dtype: Dtype,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum Dtype {
    BOOL,
    U8,
    I32,
    I64,
    F16,
    BF16,
    F32,
    F64,
}

pub struct Tensor<'a> {
    shape: Vec<usize>,
    dtype: Dtype,
    data: &'a [u8],
}

impl<'a> Tensor<'a> {
    pub fn new(data: &'a [u8], dtype: Dtype, shape: Vec<usize>) -> Self {
        Self { data, dtype, shape }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialization() {
        let data: Vec<u8> = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let attn_0 = Tensor {
            dtype: Dtype::F32,
            shape: vec![1, 2, 3],
            data: &data,
        };
        let metadata: HashMap<String, Tensor> =
            [("attn.0".to_string(), attn_0)].into_iter().collect();

        let out = SafeTensor::serialize(&metadata);
        SafeTensor::deserialize(out).unwrap();
    }

    #[test]
    fn test_gpt2() {
        gpt2_like(12, "gpt2");
    }

    #[test]
    fn test_gpt2_medium() {
        gpt2_like(24, "gpt2_medium");
    }

    fn gpt2_like(n_heads: usize, model_id: &str) {
        let mut tensors_desc = vec![];
        tensors_desc.push(("wte".to_string(), vec![50257, 768]));
        tensors_desc.push(("wpe".to_string(), vec![1024, 768]));
        for i in 0..n_heads {
            tensors_desc.push((format!("h.{}.ln_1.weight", i), vec![768]));
            tensors_desc.push((format!("h.{}.ln_1.bias", i), vec![768]));
            tensors_desc.push((format!("h.{}.attn.bias", i), vec![1, 1, 1024, 1024]));
            tensors_desc.push((format!("h.{}.attn.c_attn.weight", i), vec![768, 2304]));
            tensors_desc.push((format!("h.{}.attn.c_attn.bias", i), vec![2304]));
            tensors_desc.push((format!("h.{}.attn.c_proj.weight", i), vec![768, 768]));
            tensors_desc.push((format!("h.{}.attn.c_proj.bias", i), vec![768]));
            tensors_desc.push((format!("h.{}.ln_2.weight", i), vec![768]));
            tensors_desc.push((format!("h.{}.ln_2.bias", i), vec![768]));
            tensors_desc.push((format!("h.{}.mlp.c_fc.weight", i), vec![768, 3072]));
            tensors_desc.push((format!("h.{}.mlp.c_fc.bias", i), vec![3072]));
            tensors_desc.push((format!("h.{}.mlp.c_proj.weight", i), vec![3072, 768]));
            tensors_desc.push((format!("h.{}.mlp.c_proj.bias", i), vec![768]));
        }
        tensors_desc.push(("ln_f.weight".to_string(), vec![768]));
        tensors_desc.push(("ln_f.bias".to_string(), vec![768]));

        let n: usize = tensors_desc
            .iter()
            .map(|item| item.1.iter().product::<usize>())
            .sum::<usize>()
            * 4; // 4 == float32
        let all_data = vec![0; n];
        let mut metadata: HashMap<String, Tensor> = HashMap::new();
        let mut offset = 0;
        for (name, shape) in tensors_desc {
            let n: usize = shape.iter().product();
            let buffer = &all_data[offset..offset + n];
            let tensor = Tensor::new(buffer, Dtype::F32, shape);
            metadata.insert(name, tensor);
            offset += n;
        }

        let filename = format!("./out_{}.bin", model_id);

        use std::time::Instant;
        let start = Instant::now();
        let out = SafeTensor::serialize(&metadata);

        let start = Instant::now();
        std::fs::write(&filename, out).unwrap();

        let start = Instant::now();
        let raw = std::fs::read(&filename).unwrap();

        let start = Instant::now();
        SafeTensor::deserialize(raw).unwrap();
    }
}
