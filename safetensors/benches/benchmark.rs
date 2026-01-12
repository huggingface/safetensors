use criterion::{criterion_group, criterion_main, Criterion};
use memmap2::MmapOptions;
use safetensors::tensor::*;
use std::collections::HashMap;
use std::fs::File;
use std::hint::black_box;
use std::io::Write;

// Returns a sample data of size 2_MB
fn get_sample_data() -> (Vec<u8>, Vec<usize>, Dtype) {
    let shape = vec![1000, 500];
    let dtype = Dtype::F32;
    let nbits = shape.iter().product::<usize>() * dtype.bitsize();
    assert!(nbits % 8 == 0);
    let n: usize = nbits / 8; // 2_000_000 bytes
    let data = vec![0; n];

    (data, shape, dtype)
}

fn create_benchmark_file(n_layers: usize) -> String {
    let (data, shape, dtype) = get_sample_data();
    let mut metadata: HashMap<String, TensorView> = HashMap::new();
    for i in 0..n_layers {
        let tensor = TensorView::new(dtype, shape.clone(), &data[..]).unwrap();
        metadata.insert(format!("weight{i}"), tensor);
    }

    let out = serialize(&metadata, None).unwrap();
    let filename = format!("bench_{}.safetensors", n_layers);
    let mut file = File::create(&filename).unwrap();
    file.write_all(&out).unwrap();
    filename
}

pub fn bench_serialize(c: &mut Criterion) {
    let (data, shape, dtype) = get_sample_data();
    let n_layers = 5;

    let mut metadata: HashMap<String, TensorView> = HashMap::new();
    // 2_MB x 5 = 10_MB
    for i in 0..n_layers {
        let tensor = TensorView::new(dtype, shape.clone(), &data[..]).unwrap();
        metadata.insert(format!("weight{i}"), tensor);
    }

    c.bench_function("Serialize 10_MB", |b| {
        b.iter(|| {
            let _serialized = serialize(black_box(&metadata), black_box(None));
        })
    });
}

pub fn bench_deserialize(c: &mut Criterion) {
    let (data, shape, dtype) = get_sample_data();
    let n_layers = 5;

    let mut metadata: HashMap<String, TensorView> = HashMap::new();
    // 2_MB x 5 = 10_MB
    for i in 0..n_layers {
        let tensor = TensorView::new(dtype, shape.clone(), &data[..]).unwrap();
        metadata.insert(format!("weight{i}"), tensor);
    }

    let out = serialize(&metadata, None).unwrap();

    c.bench_function("Deserialize 10_MB", |b| {
        b.iter(|| {
            let _deserialized = SafeTensors::deserialize(black_box(&out)).unwrap();
        })
    });
}

pub fn bench_file_loading(c: &mut Criterion) {
    // 100 MB file (50 layers * 2 MB)
    let n_layers = 50;
    let filename = create_benchmark_file(n_layers);

    let mut group = c.benchmark_group("File Loading 100_MB");

    group.bench_function("mmap", |b| {
        b.iter(|| {
            let file = File::open(&filename).unwrap();
            let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
            let _safetensors = SafeTensors::deserialize(&mmap).unwrap();
        })
    });

    #[cfg(all(
        target_os = "linux",
        any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "riscv64",
            target_arch = "loongarch64",
            target_arch = "powerpc64"
        )
    ))]
    {
        group.bench_function("io_uring", |b| {
            b.iter(|| {
                let buffer = deserialize_from_file_io_uring(&filename).unwrap();
                let _safetensors = SafeTensors::deserialize(&buffer).unwrap();
            })
        });
    }

    group.finish();
    std::fs::remove_file(filename).ok();
}

criterion_group!(bench_ser, bench_serialize);
criterion_group!(bench_de, bench_deserialize);
criterion_group!(bench_load, bench_file_loading);
criterion_main!(bench_ser, bench_de, bench_load);
