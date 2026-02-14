mod bench_utils;

use bench_utils::create_test_safetensors;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use safetensors::loader::{Backend, Device, FileLoader, PrefetchConfig};
use safetensors::tensor::Dtype;
use std::hint::black_box;

fn bench_loader_open(c: &mut Criterion) {
    let (file, _) = create_test_safetensors(10, &[1000, 500], Dtype::F32);

    c.bench_function("FileLoader::open (CPU, 20MB)", |b| {
        b.iter(|| {
            let _loader = FileLoader::open(black_box(file.path()), Device::Cpu).unwrap();
        })
    });

    c.bench_function("FileLoader::with_backend (Mmap, 20MB)", |b| {
        b.iter(|| {
            let _loader =
                FileLoader::with_backend(black_box(file.path()), Device::Cpu, Backend::Mmap).unwrap();
        })
    });
}

fn bench_loader_fetch(c: &mut Criterion) {
    let mut group = c.benchmark_group("FileLoader::fetch");

    for &size_kb in &[4, 64, 1024, 4096] {
        let n_elems = size_kb * 1024 / 4; // F32 = 4 bytes
        let shape = vec![n_elems];
        let (file, infos) = create_test_safetensors(1, &shape, Dtype::F32);
        let loader = FileLoader::open(file.path(), Device::Cpu).unwrap();
        let start = infos[0].start;
        let end = infos[0].end;
        let total_bytes = end - start;

        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("mmap", format!("{size_kb}KB")),
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

fn bench_loader_fetch_view(c: &mut Criterion) {
    let mut group = c.benchmark_group("FileLoader::fetch_view");

    for &size_kb in &[4, 64, 1024, 4096] {
        let n_elems = size_kb * 1024 / 4;
        let shape = vec![n_elems];
        let (file, infos) = create_test_safetensors(1, &shape, Dtype::F32);
        let loader = FileLoader::open(file.path(), Device::Cpu).unwrap();
        let start = infos[0].start;
        let end = infos[0].end;
        let total_bytes = end - start;

        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("zero_copy", format!("{size_kb}KB")),
            &(start, end),
            |b, &(s, e)| {
                b.iter(|| {
                    let _buf = loader.fetch_view(black_box(s), black_box(e)).unwrap();
                })
            },
        );
    }

    group.finish();
}

fn bench_loader_fetch_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("FileLoader::fetch_batch");

    for &n_tensors in &[10, 50, 200] {
        // Each tensor ~ 2MB (500x1000 F32)
        let shape = vec![500, 1000];
        let (file, infos) = create_test_safetensors(n_tensors, &shape, Dtype::F32);
        let loader = FileLoader::open(file.path(), Device::Cpu).unwrap();
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

fn bench_loader_prefetch(c: &mut Criterion) {
    let mut group = c.benchmark_group("FileLoader::iter_prefetch");

    for &n_tensors in &[10, 50, 200] {
        let shape = vec![500, 1000]; // 2MB per tensor
        let (file, infos) = create_test_safetensors(n_tensors, &shape, Dtype::F32);
        let loader = FileLoader::open(file.path(), Device::Cpu).unwrap();
        let total_bytes: usize = infos.iter().map(|t| t.size()).sum();

        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("prefetch_4", format!("{n_tensors}x2MB")),
            &infos,
            |b, infos| {
                b.iter(|| {
                    let results: Vec<_> = loader
                        .iter_prefetch(black_box(infos), PrefetchConfig::new(4))
                        .collect();
                    black_box(results);
                })
            },
        );
    }

    group.finish();
}

fn bench_fetch_vs_view(c: &mut Criterion) {
    let mut group = c.benchmark_group("fetch_vs_view_4MB");

    let shape = vec![1024, 1024]; // 4MB (F32)
    let (file, infos) = create_test_safetensors(1, &shape, Dtype::F32);
    let start = infos[0].start;
    let end = infos[0].end;
    let total_bytes = end - start;

    let loader = FileLoader::open(file.path(), Device::Cpu).unwrap();
    group.throughput(Throughput::Bytes(total_bytes as u64));

    group.bench_function("fetch (copy)", |b| {
        b.iter(|| {
            let _buf = loader.fetch(black_box(start), black_box(end)).unwrap();
        })
    });

    group.bench_function("fetch_view (zero-copy)", |b| {
        b.iter(|| {
            let _buf = loader
                .fetch_view(black_box(start), black_box(end))
                .unwrap();
        })
    });

    group.bench_function("fetch_to_vec (copy+own)", |b| {
        b.iter(|| {
            let _vec = loader
                .fetch_to_vec(black_box(start), black_box(end))
                .unwrap();
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_loader_open,
    bench_loader_fetch,
    bench_loader_fetch_view,
    bench_loader_fetch_batch,
    bench_loader_prefetch,
    bench_fetch_vs_view,
);
criterion_main!(benches);
