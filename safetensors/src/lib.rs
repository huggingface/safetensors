use serde::{Deserialize, Serialize};
use std::collections::HashMap;

enum SafeTensorError {
    InvalidHeader,
}

struct SafeTensor {
    metadata: Metadata,
    data: Vec<u8>,
}

impl SafeTensor {
    pub fn serialize(data: &HashMap<String, OwnedTensor>) -> Vec<u8> {
        let mut tensors: Vec<&OwnedTensor> = vec![];
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
            buffer.extend(&tensor.data);
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
        let data = &buffer[8 + n..];
        Ok(SafeTensor {
            metadata,
            data: buffer,
        })
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct Metadata(HashMap<String, TensorInfo>);

#[derive(Debug, Deserialize, Serialize, Clone)]
struct TensorInfo {
    dtype: Dtype,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
enum Dtype {
    BOOL,
    U8,
    F16,
    BF16,
    F32,
    F64,
}

struct Tensor<'a> {
    shape: Vec<usize>,
    dtype: Dtype,
    data: &'a [u8],
}

struct OwnedTensor {
    shape: Vec<usize>,
    dtype: Dtype,
    data: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialization() {
        let attn_0 = OwnedTensor {
            dtype: Dtype::F32,
            shape: vec![1, 2, 3],
            data: vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]
                .into_iter()
                .flat_map(|f| f.to_le_bytes())
                .collect(),
        };
        let metadata: HashMap<String, OwnedTensor> =
            [("attn.0".to_string(), attn_0)].into_iter().collect();

        let out = SafeTensor::serialize(&metadata);
        let loaded = SafeTensor::deserialize(out);
    }

    fn get_tensor(shape: Vec<usize>) -> OwnedTensor {
        let n = shape.iter().product();
        let data = vec![0.0f32; n];
        OwnedTensor {
            dtype: Dtype::F32,
            shape,
            data: data.into_iter().flat_map(|f| f.to_le_bytes()).collect(),
        }
    }

    #[test]
    fn test_gpt2() {
        gpt2_like(12);
    }

    #[test]
    fn test_gpt2_medium() {
        gpt2_like(24);
    }

    fn gpt2_like(n_heads: usize) {
        let mut tensors = vec![];
        tensors.push(("wte".to_string(), get_tensor(vec![50257, 768])));
        tensors.push(("wpe".to_string(), get_tensor(vec![1024, 768])));
        for i in 0..n_heads {
            tensors.push((format!("h.{}.ln_1.weight", i), get_tensor(vec![768])));
            tensors.push((format!("h.{}.ln_1.bias", i), get_tensor(vec![768])));
            tensors.push((
                format!("h.{}.attn.bias", i),
                get_tensor(vec![1, 1, 1024, 1024]),
            ));
            tensors.push((
                format!("h.{}.attn.c_attn.weight", i),
                get_tensor(vec![768, 2304]),
            ));
            tensors.push((format!("h.{}.attn.c_attn.bias", i), get_tensor(vec![2304])));
            tensors.push((
                format!("h.{}.attn.c_proj.weight", i),
                get_tensor(vec![768, 768]),
            ));
            tensors.push((format!("h.{}.attn.c_proj.bias", i), get_tensor(vec![768])));
            tensors.push((format!("h.{}.ln_2.weight", i), get_tensor(vec![768])));
            tensors.push((format!("h.{}.ln_2.bias", i), get_tensor(vec![768])));
            tensors.push((
                format!("h.{}.mlp.c_fc.weight", i),
                get_tensor(vec![768, 3072]),
            ));
            tensors.push((format!("h.{}.mlp.c_fc.bias", i), get_tensor(vec![3072])));
            tensors.push((
                format!("h.{}.mlp.c_proj.weight", i),
                get_tensor(vec![3072, 768]),
            ));
            tensors.push((format!("h.{}.mlp.c_proj.bias", i), get_tensor(vec![768])));
        }
        tensors.push(("ln_f.weight".to_string(), get_tensor(vec![768])));
        tensors.push(("ln_f.bias".to_string(), get_tensor(vec![768])));
        let metadata: HashMap<String, OwnedTensor> = tensors.into_iter().collect();

        use std::time::Instant;
        let start = Instant::now();
        let out = SafeTensor::serialize(&metadata);
        println!("Serialization {:?}", start.elapsed());

        let start = Instant::now();
        std::fs::write("./out.bin", out);
        println!("Write to file {:?}", start.elapsed());

        let start = Instant::now();
        let raw = std::fs::read("./out.bin").unwrap();
        println!("Read from file {:?}", start.elapsed());

        let start = Instant::now();
        let raw2 = std::fs::read("./out.bin").unwrap();
        println!("Read twice from file {:?}", start.elapsed());

        let start = Instant::now();
        let loaded = SafeTensor::deserialize(raw2);
        println!("Deserialization {:?}", start.elapsed());
    }

    // use std::fs::File;
    // use std::io::Read;
    // use std::time::Instant;
    // #[test]
    // fn read_gpt2() {
    //     let start = Instant::now();
    //     let filename = "gpt2.bin";
    //     let mut f = File::open(&filename).unwrap();
    //     let metadata = std::fs::metadata(&filename).expect("unable to read metadata");
    //     let mut buffer = vec![0; metadata.len() as usize];
    //     f.read(&mut buffer).expect("buffer overflow");
    //     println!("Read gpt2 in {:?}", start.elapsed());
    // }
}
