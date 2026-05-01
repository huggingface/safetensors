#![no_main]

use arbitrary::Unstructured;
use libfuzzer_sys::fuzz_target;
use safetensors::tensor::{serialize, Dtype, SafeTensors, View};
use std::borrow::Cow;
use std::collections::HashMap;

const MAX_TENSORS: u8 = 8;
const MAX_RANK: u8 = 4;
const MAX_DIM: u8 = 8;
const MAX_TENSOR_BYTES: usize = 4096;
const MAX_TOTAL_BYTES: usize = 16 * 1024;

#[derive(Debug)]
struct OwnedTensor {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl View for &OwnedTensor {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        self.data.as_slice().into()
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

fuzz_target!(|data: &[u8]| {
    let mut input = Unstructured::new(data);
    let Ok((tensors, metadata)) = build_input(&mut input) else {
        return;
    };

    let Ok(serialized) = serialize(
        tensors.iter().map(|tensor| (tensor.name.as_str(), tensor)),
        metadata,
    ) else {
        return;
    };

    let parsed = SafeTensors::deserialize(&serialized).expect("serialized tensors should parse");
    assert_eq!(parsed.len(), tensors.len());

    for tensor in tensors {
        let parsed_tensor = parsed
            .tensor(&tensor.name)
            .expect("serialized tensor name should be present");
        assert_eq!(parsed_tensor.dtype(), tensor.dtype);
        assert_eq!(parsed_tensor.shape(), tensor.shape.as_slice());
        assert_eq!(parsed_tensor.data(), tensor.data.as_slice());
    }
});

fn build_input(
    input: &mut Unstructured<'_>,
) -> arbitrary::Result<(Vec<OwnedTensor>, Option<HashMap<String, String>>)> {
    let tensor_count = input.int_in_range(0..=MAX_TENSORS)?;
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    let mut total_bytes = 0usize;

    for index in 0..tensor_count {
        let dtype = arbitrary_dtype(input)?;
        let shape = arbitrary_shape(input)?;
        let data_len = tensor_data_len(dtype, &shape).ok_or(arbitrary::Error::IncorrectFormat)?;

        if data_len > MAX_TENSOR_BYTES || total_bytes.saturating_add(data_len) > MAX_TOTAL_BYTES {
            return Err(arbitrary::Error::IncorrectFormat);
        }

        let data = input.bytes(data_len)?.to_vec();
        total_bytes += data_len;
        tensors.push(OwnedTensor {
            name: format!("tensor_{index}"),
            dtype,
            shape,
            data,
        });
    }

    let metadata = if input.arbitrary::<bool>()? {
        Some(arbitrary_metadata(input)?)
    } else {
        None
    };

    Ok((tensors, metadata))
}

fn arbitrary_dtype(input: &mut Unstructured<'_>) -> arbitrary::Result<Dtype> {
    Ok(match input.int_in_range(0..=20)? {
        0 => Dtype::BOOL,
        1 => Dtype::F4,
        2 => Dtype::F6_E2M3,
        3 => Dtype::F6_E3M2,
        4 => Dtype::U8,
        5 => Dtype::I8,
        6 => Dtype::F8_E5M2,
        7 => Dtype::F8_E4M3,
        8 => Dtype::F8_E8M0,
        9 => Dtype::F8_E4M3FNUZ,
        10 => Dtype::F8_E5M2FNUZ,
        11 => Dtype::I16,
        12 => Dtype::U16,
        13 => Dtype::F16,
        14 => Dtype::BF16,
        15 => Dtype::I32,
        16 => Dtype::U32,
        17 => Dtype::F32,
        18 => Dtype::C64,
        19 => Dtype::F64,
        _ => Dtype::I64,
    })
}

fn arbitrary_shape(input: &mut Unstructured<'_>) -> arbitrary::Result<Vec<usize>> {
    let rank = input.int_in_range(0..=MAX_RANK)?;
    let mut shape = Vec::with_capacity(rank as usize);
    for _ in 0..rank {
        shape.push(input.int_in_range(0..=MAX_DIM)? as usize);
    }
    Ok(shape)
}

fn tensor_data_len(dtype: Dtype, shape: &[usize]) -> Option<usize> {
    let elements = shape
        .iter()
        .copied()
        .try_fold(1usize, usize::checked_mul)?;
    let bits = elements.checked_mul(dtype.bitsize())?;
    if bits % 8 == 0 {
        Some(bits / 8)
    } else {
        None
    }
}

fn arbitrary_metadata(
    input: &mut Unstructured<'_>,
) -> arbitrary::Result<HashMap<String, String>> {
    let count = input.int_in_range(0..=4)?;
    let mut metadata = HashMap::with_capacity(count as usize);
    for index in 0..count {
        metadata.insert(
            format!("key_{index}_{}", arbitrary_ascii(input, 8)?),
            arbitrary_ascii(input, 16)?,
        );
    }
    Ok(metadata)
}

fn arbitrary_ascii(input: &mut Unstructured<'_>, max_len: usize) -> arbitrary::Result<String> {
    let len = input.int_in_range(0..=max_len)?;
    let bytes = input.bytes(len)?;
    Ok(bytes
        .iter()
        .map(|byte| char::from(b'a' + (byte % 26)))
        .collect())
}
