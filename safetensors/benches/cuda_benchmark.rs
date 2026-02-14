mod bench_utils;

use bench_utils::create_test_safetensors;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use safetensors::loader::{Backend, Device, FileLoader};
use safetensors::tensor::Dtype;
use std::hint::black_box;

fn bench_cuda_fetch(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda_fetch");

    for &size_kb in &[64, 1024, 4096, 16384] {
        let n_elems = size_kb * 1024 / 4; // F32 = 4 bytes
        let shape = vec![n_elems];
        let (file, infos) = create_test_safetensors(1, &shape, Dtype::F32);
        let loader = FileLoader::open(file.path(), Device::Cuda(0)).unwrap();
        let start = infos[0].start;
        let end = infos[0].end;
        let total_bytes = end - start;

        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("cuda0", format!("{size_kb}KB")),
            &(start, end),
            |b, &(s, e)| {
                b.iter(|| {
                    let _buf = loader.fetch(black_box(s), black_box(e)).unwrap();
                })
            },
        );
    }

    group.finish();
}

fn bench_cuda_fetch_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda_fetch_batch");

    for &n_tensors in &[10, 50, 200] {
        // Each tensor ~ 2MB (500x1000 F32)
        let shape = vec![500, 1000];
        let (file, infos) = create_test_safetensors(n_tensors, &shape, Dtype::F32);
        let loader = FileLoader::open(file.path(), Device::Cuda(0)).unwrap();
        let total_bytes: usize = infos.iter().map(|t| t.size()).sum();

        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("batch", format!("{n_tensors}x2MB")),
            &infos,
            |b, infos| {
                b.iter(|| {
                    let _results = loader.fetch_batch(black_box(infos)).unwrap();
                })
            },
        );
    }

    group.finish();
}

fn bench_cuda_mmap(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda_mmap");

    // 16MB tensor
    let shape = vec![2048, 2048]; // 16MB F32
    let (file, infos) = create_test_safetensors(1, &shape, Dtype::F32);
    let start = infos[0].start;
    let end = infos[0].end;
    let total_bytes = end - start;

    group.throughput(Throughput::Bytes(total_bytes as u64));

    let loader_mmap =
        FileLoader::with_backend(file.path(), Device::Cuda(0), Backend::Mmap).unwrap();
    group.bench_function("mmap_cuda", |b| {
        b.iter(|| {
            let _buf = loader_mmap.fetch(black_box(start), black_box(end)).unwrap();
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_cuda_fetch,
    bench_cuda_fetch_batch,
    bench_cuda_mmap,
);
criterion_main!(benches);
