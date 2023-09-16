use criterion::{black_box, criterion_group, criterion_main, Criterion};
use safetensors::tensor::*;
use std::collections::HashMap;

// Returns a sample data of size 2_MB
fn get_sample_data() -> (Vec<u8>, Vec<usize>, Dtype) {
    let shape = vec![1000, 500];
    let dtype = Dtype::F32;
    let n: usize = shape.iter().product::<usize>() * dtype.size(); // 4
    let data = vec![0; n];

    (data, shape, dtype)
}

// Return a shallow sample data of size 5 KiB
fn get_shallow_sample_data() -> (Vec<u8>, Vec<usize>, Dtype) {
    let shape = vec![32, 4];
    let dtype = Dtype::F32;
    let n: usize = shape.iter().product::<usize>() * dtype.size(); // 4
    let data = vec![0; n];

    (data, shape, dtype)
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

    c.bench_function("Serlialize 10_MB", |b| {
        b.iter(|| {
            let _serialized = serialize(black_box(&metadata), black_box(&None));
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

    let out = serialize(&metadata, &None).unwrap();

    c.bench_function("Deserlialize 10_MB", |b| {
        b.iter(|| {
            let _deserialized = SafeTensors::deserialize(black_box(&out)).unwrap();
        })
    });
}

pub fn bench_deserialize_shallow(c: &mut Criterion) {
    // Unlike classical models, LoRAs (and other derivatives) have a lot of "shallow" layers
    // where the size of the weights themselves does not matter as much as the number of layers
    // when it comes to the performance.

    let (data, shape, dtype) = get_shallow_sample_data();
    let n_layers = 4000;

    let mut metadata: HashMap<String, TensorView> = HashMap::new();
    // 5 KiB x 2000 = 10 MB
    for i in 0..n_layers {
        let tensor = TensorView::new(dtype, shape.clone(), &data[..]).unwrap();
        metadata.insert(format!("weight{i}"), tensor);
    }

    let out = serialize(&metadata, &None).unwrap();

    c.bench_function("Deserlialize 10_MB shallow", |b| {
        b.iter(|| {
            let _deserialized = SafeTensors::deserialize(black_box(&out)).unwrap();
        })
    });
}

criterion_group!(bench_ser, bench_serialize);
criterion_group!(bench_de, bench_deserialize, bench_deserialize_shallow);
criterion_main!(bench_ser, bench_de);
