use safetensors::loader::TensorLoadInfo;
use safetensors::tensor::{serialize, Dtype, TensorView};
use std::collections::HashMap;
use std::io::Write;

/// Create a test safetensors file with N tensors of a given shape.
/// Returns (temp_file, tensor_infos) where infos have file-absolute byte offsets.
pub fn create_test_safetensors(
    n_tensors: usize,
    shape: &[usize],
    dtype: Dtype,
) -> (tempfile::NamedTempFile, Vec<TensorLoadInfo>) {
    let elem_bytes = dtype.bitsize() / 8;
    let tensor_size: usize = shape.iter().product::<usize>() * elem_bytes;
    let data: Vec<u8> = (0..tensor_size).map(|i| (i % 256) as u8).collect();

    let mut metadata: HashMap<String, TensorView> = HashMap::new();
    for i in 0..n_tensors {
        let tensor = TensorView::new(dtype, shape.to_vec(), &data).unwrap();
        metadata.insert(format!("layer{i}.weight"), tensor);
    }

    let serialized = serialize(&metadata, None).unwrap();

    // Parse back to get offsets via pointer arithmetic on TensorView data slices
    let st = safetensors::SafeTensors::deserialize(&serialized).unwrap();

    let mut infos: Vec<TensorLoadInfo> = Vec::new();
    for (name, tv) in st.tensors() {
        let data_ptr = tv.data().as_ptr() as usize;
        let base_ptr = serialized.as_ptr() as usize;
        let file_start = data_ptr - base_ptr;
        let file_end = file_start + tv.data().len();
        infos.push(TensorLoadInfo::new(name, file_start, file_end));
    }
    // Sort by offset for sequential access patterns
    infos.sort_by_key(|t| t.start);

    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(&serialized).unwrap();
    file.flush().unwrap();

    (file, infos)
}
