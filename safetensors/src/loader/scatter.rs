//! Multi-device scatter loading for distributing model shards across GPUs.
//!
//! This module provides optimized strategies for loading sharded safetensors models
//! onto multiple CUDA devices, minimizing NVMe reads by reading each shard once
//! and scattering to all target devices.
//!
//! # Strategies
//!
//! - **io_uring scatter**: O_DIRECT reads to pinned staging, then async cudaMemcpy
//!   to each target GPU (selective byte ranges). ~69-70 GB/s on NVMe RAID0.
//! - **cuFile P2P scatter**: cuFileRead to a primary GPU, then D2D async copies to
//!   secondaries. ~30 GB/s.
//! - **Single-device cuFile**: Parallel cuFileRead per shard to one GPU.
//! - **Default**: Parallel FileLoader::with_backend + optional preload.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use super::{Backend, Buffer, Device, FileLoader, LoaderError, Result};

/// Per-shard specification for scatter loading.
pub struct ShardSpec {
    /// Path to the safetensors shard file.
    pub path: PathBuf,
    /// Header size (byte offset where tensor data begins).
    pub offset: usize,
    /// Per-device sorted tensor byte ranges: device_index → Vec<(start, end)>.
    /// Only populated for multi-device CUDA loads.
    pub dev_ranges: HashMap<usize, Vec<(usize, usize)>>,
}

/// Per-shard result from scatter loading.
pub struct ShardLoadResult {
    /// Loader for on-demand tensor access (mmap-backed for multi-device, or primary loader).
    pub loader: Arc<FileLoader>,
    /// Single-device preloaded buffer (entire shard data section on one GPU).
    pub preloaded: Option<Arc<Buffer>>,
    /// Multi-device scattered buffers: device_index → GPU buffer with full shard data.
    /// Used by cuFile P2P scatter (each GPU gets complete shard data section).
    pub per_device: Option<HashMap<usize, Arc<Buffer>>>,
    /// Per-tensor GPU buffers from io_uring scatter.
    /// Maps (data_section_offset, device_idx) → individual GPU buffer sized exactly
    /// for that tensor's data. Offset into each buffer is always 0.
    pub per_tensor: Option<HashMap<(usize, usize), Arc<Buffer>>>,
}

/// A tensor that has been fully scattered to its target GPU.
/// Emitted by the streaming callback during io_uring scatter.
pub struct TensorReady {
    /// Index of the shard this tensor came from.
    pub shard_idx: usize,
    /// Data-section byte offset where this tensor starts (lookup key for tensor metadata).
    pub data_offset: usize,
    /// Target GPU device index.
    pub device_idx: usize,
    /// GPU buffer containing this tensor's data.
    pub buffer: Arc<Buffer>,
}

// SAFETY: TensorReady can be sent across threads.
// The Arc<Buffer> holds a CUDA device pointer which is valid from any thread.
unsafe impl Send for TensorReady {}

/// Configuration for scatter loading.
pub struct ScatterConfig {
    /// Backend to use for I/O.
    pub backend: Backend,
    /// Target device (Cuda(idx) for single-device, ignored for multi-device).
    pub device: Device,
    /// True if all tensors go to one device (no scatter needed).
    pub is_single: bool,
}

/// Cached pinned staging buffer for multi-device io_uring scatter.
/// Reused across `safe_open()` calls to avoid the ~28ms `cudaHostAlloc` overhead.
/// The buffer is device-agnostic host pinned memory (CUDA_HOST_ALLOC_PORTABLE).
#[cfg(all(target_os = "linux", feature = "io_uring", feature = "cuda"))]
static STAGING_CACHE: std::sync::Mutex<Option<StagingCache>> = std::sync::Mutex::new(None);

#[cfg(all(target_os = "linux", feature = "io_uring", feature = "cuda"))]
struct StagingCache {
    base_ptr: usize,
    total_size: usize,
}

/// Free the cached pinned staging buffer.
///
/// Call this to release the pinned host memory held by the staging cache.
/// The buffer is automatically reused across scatter_load() calls, so this
/// is only needed for explicit memory management.
#[cfg(all(target_os = "linux", feature = "io_uring", feature = "cuda"))]
pub fn clear_staging_cache() {
    use super::cuda::cuda_free_host;
    let old_ptr = {
        let mut cache = STAGING_CACHE.lock().unwrap();
        let ptr = cache.as_ref().map(|c| c.base_ptr);
        *cache = None;
        ptr
    };
    if let Some(ptr) = old_ptr {
        cuda_free_host(ptr as *mut u8);
    }
}

/// No-op when io_uring + CUDA are not available.
#[cfg(not(all(target_os = "linux", feature = "io_uring", feature = "cuda")))]
pub fn clear_staging_cache() {}

/// Collect per-shard device sets and the sorted list of unique devices across all shards.
#[cfg(all(
    feature = "cuda",
    any(
        all(target_os = "linux", feature = "cufile"),
        all(target_os = "linux", feature = "io_uring"),
    )
))]
fn collect_device_sets(specs: &[ShardSpec]) -> (Vec<std::collections::HashSet<usize>>, Vec<usize>) {
    let shard_devices: Vec<std::collections::HashSet<usize>> = specs
        .iter()
        .map(|spec| spec.dev_ranges.keys().copied().collect())
        .collect();

    let mut unique_devices: Vec<usize> = shard_devices
        .iter()
        .flat_map(|s| s.iter().copied())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    unique_devices.sort();

    (shard_devices, unique_devices)
}

/// Load multiple shards, picking the optimal strategy.
///
/// Dispatches to the best available loader based on the configuration:
/// - Multi-device + cuFile → cuFile P2P scatter
/// - Multi-device + other → io_uring scatter
/// - Single-device + cuFile → parallel cuFileRead
/// - Default → parallel FileLoader::with_backend + optional preload
pub fn scatter_load(
    specs: &[ShardSpec],
    shard_paths: &[PathBuf],
    shard_offsets: &[usize],
    config: &ScatterConfig,
) -> Result<Vec<ShardLoadResult>> {
    let _ = specs; // used only by CUDA scatter strategies

    #[cfg(feature = "cuda")]
    let is_cuda = matches!(config.device, Device::Cuda(_));
    #[cfg(not(feature = "cuda"))]
    let is_cuda = false;

    #[cfg(all(target_os = "linux", feature = "cufile"))]
    let is_cufile = matches!(config.backend, Backend::CuFile);
    #[cfg(not(all(target_os = "linux", feature = "cufile")))]
    let is_cufile = false;

    if is_cuda && !config.is_single && is_cufile {
        #[cfg(all(target_os = "linux", feature = "cufile", feature = "cuda"))]
        return scatter_cufile_p2p(specs, shard_paths, shard_offsets, config);
        #[cfg(not(all(target_os = "linux", feature = "cufile", feature = "cuda")))]
        unreachable!()
    } else if is_cuda && !config.is_single && !is_cufile {
        #[cfg(all(target_os = "linux", feature = "io_uring", feature = "cuda"))]
        return scatter_iouring(specs, shard_paths, shard_offsets, config, None);
        #[cfg(not(all(target_os = "linux", feature = "io_uring", feature = "cuda")))]
        return load_default(shard_paths, shard_offsets, config);
    } else if is_cuda && config.is_single && is_cufile {
        #[cfg(all(target_os = "linux", feature = "cufile", feature = "cuda"))]
        return load_single_cufile(shard_paths, shard_offsets, config);
        #[cfg(not(all(target_os = "linux", feature = "cufile", feature = "cuda")))]
        unreachable!()
    } else {
        load_default(shard_paths, shard_offsets, config)
    }
}

/// Start a streaming io_uring scatter in a background thread.
///
/// Returns a bounded channel receiver that yields `TensorReady` results as each
/// tensor completes its GPU transfer, and a JoinHandle whose value contains the
/// `ShardLoadResult` vec (for populating per_tensor_buffers afterward).
///
/// Back-pressure: `SyncSender::send()` blocks when the channel is full, pausing
/// io_uring reads naturally. No deadlock because the consumer drains the channel.
#[cfg(all(target_os = "linux", feature = "io_uring", feature = "cuda"))]
pub fn scatter_iouring_start(
    specs: Vec<ShardSpec>,
    shard_paths: Vec<PathBuf>,
    shard_offsets: Vec<usize>,
    config: ScatterConfig,
    prefetch_count: usize,
) -> (
    std::sync::mpsc::Receiver<std::result::Result<TensorReady, String>>,
    std::thread::JoinHandle<Result<Vec<ShardLoadResult>>>,
) {
    let (tx, rx) = std::sync::mpsc::sync_channel(prefetch_count);

    let handle = std::thread::spawn(move || {
        let on_ready = |ready: TensorReady| {
            // Ignore send errors — consumer may have dropped the receiver
            let _ = tx.send(Ok(ready));
        };
        let result = scatter_iouring(
            &specs,
            &shard_paths,
            &shard_offsets,
            &config,
            Some(&on_ready),
        );
        if let Err(ref e) = result {
            let _ = tx.send(Err(format!("{e}")));
        }
        // tx drops here, closing the channel
        result
    });

    (rx, handle)
}

/// Multi-device cuFile: read each shard ONCE to a "primary" GPU via cuFile,
/// then P2P scatter tensor byte ranges to other GPUs via cudaMemcpyAsync D2D.
#[cfg(all(target_os = "linux", feature = "cufile", feature = "cuda"))]
fn scatter_cufile_p2p(
    specs: &[ShardSpec],
    shard_paths: &[PathBuf],
    shard_offsets: &[usize],
    config: &ScatterConfig,
) -> Result<Vec<ShardLoadResult>> {
    use super::cuda::{
        cuda_alloc_buffer, cuda_enable_peer_access, cuda_memcpy_dtod_async, cuda_stream_create,
        cuda_stream_destroy, cuda_stream_sync, set_cuda_device,
    };

    struct Entry {
        shard_idx: usize,
        device_idx: usize,
        loader: Option<Arc<FileLoader>>,
        buffer: Buffer,
        offset: usize,
        is_primary: bool,
    }
    unsafe impl Send for Entry {}
    unsafe impl Sync for Entry {}

    let (shard_devices, unique_devices) = collect_device_sets(specs);

    // For each shard, pick primary = device needing the most bytes
    let shard_primary: Vec<Option<usize>> = specs
        .iter()
        .map(|spec| {
            spec.dev_ranges
                .iter()
                .map(|(&dev_idx, ranges)| {
                    let total: usize = ranges.iter().map(|(s, e)| e - s).sum();
                    (dev_idx, total)
                })
                .max_by_key(|&(_, total)| total)
                .map(|(dev_idx, _)| dev_idx)
        })
        .collect();

    // Enable P2P access between all device pairs (for D2D copies)
    for &dev_a in &unique_devices {
        set_cuda_device(dev_a)?;
        for &dev_b in &unique_devices {
            if dev_a != dev_b {
                cuda_enable_peer_access(dev_b)?;
            }
        }
    }

    let backend = config.backend;
    let shard_devices = &shard_devices;
    let shard_primary = &shard_primary;

    // Phase 1: Allocate GPU buffers (parallel by device).
    let entries: Vec<Entry> = std::thread::scope(|s| {
        let handles: Vec<_> = unique_devices
            .iter()
            .map(|&dev_idx| {
                s.spawn(move || -> std::result::Result<Vec<Entry>, String> {
                    set_cuda_device(dev_idx)
                        .map_err(|e| format!("cudaSetDevice({dev_idx}): {e}"))?;
                    let mut device_entries = Vec::new();
                    for (shard_idx, (path, &offset)) in
                        shard_paths.iter().zip(shard_offsets.iter()).enumerate()
                    {
                        if !shard_devices[shard_idx].contains(&dev_idx) {
                            continue;
                        }
                        let is_primary = shard_primary[shard_idx] == Some(dev_idx);
                        let file_size = std::fs::metadata(path)
                            .map_err(|e| format!("stat: {e}"))?
                            .len() as usize;
                        let data_size = file_size - offset;

                        let buffer = cuda_alloc_buffer(data_size).map_err(|e| {
                            format!("cudaMalloc shard {shard_idx} dev {dev_idx}: {e}")
                        })?;

                        let loader = if is_primary {
                            let l = FileLoader::with_backend(path, Device::Cuda(dev_idx), backend)
                                .map_err(|e| {
                                    format!("loader shard {shard_idx} dev {dev_idx}: {e}")
                                })?;
                            super::cufile_buf_register(buffer.as_ptr(), buffer.len());
                            Some(Arc::new(l))
                        } else {
                            None
                        };

                        device_entries.push(Entry {
                            shard_idx,
                            device_idx: dev_idx,
                            loader,
                            buffer,
                            offset,
                            is_primary,
                        });
                    }
                    Ok(device_entries)
                })
            })
            .collect();

        let mut entries = Vec::new();
        for handle in handles {
            let device_entries = handle
                .join()
                .map_err(|_| LoaderError::FetchError("Thread panicked".to_string()))?
                .map_err(LoaderError::FetchError)?;
            entries.extend(device_entries);
        }
        Ok::<Vec<Entry>, LoaderError>(entries)
    })?;

    // Build shard → entry indices for Phase 2
    let mut shard_entry_map: Vec<Vec<usize>> = vec![Vec::new(); shard_paths.len()];
    for (i, entry) in entries.iter().enumerate() {
        shard_entry_map[entry.shard_idx].push(i);
    }

    // Phase 2: Per-shard parallel — cuFile read to primary, then P2P scatter.
    let entries_ref = &entries;
    let shard_entry_ref = &shard_entry_map;

    std::thread::scope(|s| {
        let handles: Vec<_> = shard_entry_ref
            .iter()
            .enumerate()
            .filter(|(_, idxs)| !idxs.is_empty())
            .map(|(shard_idx, entry_indices)| {
                s.spawn(move || -> std::result::Result<(), String> {
                    // Find primary entry
                    let primary_ei = *entry_indices
                        .iter()
                        .find(|&&i| entries_ref[i].is_primary)
                        .ok_or_else(|| format!("No primary for shard {shard_idx}"))?;
                    let primary = &entries_ref[primary_ei];

                    // cuFile read to primary GPU
                    let loader = primary.loader.as_ref().unwrap();
                    loader
                        .read_into(&primary.buffer, primary.offset)
                        .map_err(|e| {
                            format!(
                                "cuFileRead shard {shard_idx} dev {}: {e}",
                                primary.device_idx
                            )
                        })?;

                    // P2P scatter to secondaries
                    for &ei in entry_indices {
                        if ei == primary_ei {
                            continue;
                        }
                        let secondary = &entries_ref[ei];
                        let ranges = match specs[shard_idx].dev_ranges.get(&secondary.device_idx) {
                            Some(r) if !r.is_empty() => r,
                            _ => continue,
                        };

                        set_cuda_device(secondary.device_idx)
                            .map_err(|e| format!("cudaSetDevice({}): {e}", secondary.device_idx))?;
                        let stream =
                            cuda_stream_create().map_err(|e| format!("cudaStreamCreate: {e}"))?;

                        for &(start, end) in ranges {
                            let size = end - start;
                            if size == 0 {
                                continue;
                            }
                            unsafe {
                                cuda_memcpy_dtod_async(
                                    (secondary.buffer.as_ptr() as *mut u8).add(start),
                                    primary.buffer.as_ptr().add(start),
                                    size,
                                    stream,
                                )
                                .map_err(|e| {
                                    format!(
                                        "D2D shard {shard_idx} {}→{}: {e}",
                                        primary.device_idx, secondary.device_idx
                                    )
                                })?;
                            }
                        }

                        unsafe { cuda_stream_sync(stream) }
                            .map_err(|e| format!("stream sync: {e}"))?;
                        unsafe { cuda_stream_destroy(stream) };
                    }
                    Ok(())
                })
            })
            .collect();

        for handle in handles {
            handle
                .join()
                .map_err(|_| LoaderError::FetchError("Thread panicked".to_string()))?
                .map_err(LoaderError::FetchError)?;
        }
        Ok::<(), LoaderError>(())
    })?;

    // Deregister primary cuFile buffers and build result maps
    let mut shard_default_loaders: Vec<Option<Arc<FileLoader>>> = vec![None; shard_paths.len()];
    let mut per_device_maps: Vec<HashMap<usize, Arc<Buffer>>> =
        vec![HashMap::new(); shard_paths.len()];
    for entry in entries {
        if entry.is_primary {
            if shard_default_loaders[entry.shard_idx].is_none() {
                shard_default_loaders[entry.shard_idx] = entry.loader.clone();
            }
            super::cufile_buf_deregister(entry.buffer.as_ptr());
        }
        per_device_maps[entry.shard_idx].insert(entry.device_idx, Arc::new(entry.buffer));
    }

    for (shard_idx, path) in shard_paths.iter().enumerate() {
        if shard_default_loaders[shard_idx].is_none() {
            let loader = FileLoader::with_backend(path, Device::Cpu, Backend::Mmap).map_err(|e| {
                LoaderError::FetchError(format!("loader for {}: {e}", path.display()))
            })?;
            shard_default_loaders[shard_idx] = Some(Arc::new(loader));
        }
    }

    Ok(shard_default_loaders
        .into_iter()
        .zip(per_device_maps)
        .map(|(loader_opt, per_device_map)| {
            let loader = loader_opt.unwrap();
            let per_device = if per_device_map.is_empty() {
                None
            } else {
                Some(per_device_map)
            };
            ShardLoadResult {
                loader,
                preloaded: None,
                per_device,
                per_tensor: None,
            }
        })
        .collect())
}

/// Multi-device io_uring: read each shard ONCE from NVMe via O_DIRECT,
/// then scatter chunks to all target GPUs via cudaMemcpyAsync.
#[cfg(all(target_os = "linux", feature = "io_uring", feature = "cuda"))]
fn scatter_iouring(
    specs: &[ShardSpec],
    shard_paths: &[PathBuf],
    shard_offsets: &[usize],
    _config: &ScatterConfig,
    on_ready: Option<&(dyn Fn(TensorReady) + Send + Sync)>,
) -> Result<Vec<ShardLoadResult>> {
    use super::cuda::{
        cuda_alloc_buffer, cuda_free_host, cuda_host_alloc, cuda_memcpy_htod_async,
        cuda_stream_create, cuda_stream_destroy, cuda_stream_sync, set_cuda_device,
    };

    let (shard_devices, unique_devices) = collect_device_sets(specs);

    // Constants for io_uring scatter.
    const CHUNK_SIZE: usize = 4 << 20; // 4 MB per staging buffer slot
    const N_STAGE: usize = 4; // io_uring QD per ring
    const ALIGN: usize = 4096; // O_DIRECT alignment

    struct GpuAlloc {
        shard_idx: usize,
        device_idx: usize,
        buffers: Vec<Arc<Buffer>>, // one per range in dev_ranges (Arc for streaming callback)
        range_starts: Vec<usize>,  // data-section start offset per range (parallel to buffers)
    }
    unsafe impl Send for GpuAlloc {}
    unsafe impl Sync for GpuAlloc {}

    // Per-shard sorted target device lists
    let shard_target_devices: Vec<Vec<usize>> = shard_devices
        .iter()
        .map(|devs| {
            let mut v: Vec<usize> = devs.iter().copied().collect();
            v.sort();
            v
        })
        .collect();

    let shard_devices = &shard_devices;
    let shard_target_devices = &shard_target_devices;

    let active_shards: Vec<(usize, &std::path::PathBuf, usize)> = shard_paths
        .iter()
        .zip(shard_offsets.iter())
        .enumerate()
        .filter(|(si, _)| !shard_devices[*si].is_empty())
        .map(|(si, (path, &offset))| (si, path, offset))
        .collect();
    let n_active = active_shards.len();
    let active_shards = &active_shards;

    let total_staging = n_active * N_STAGE * CHUNK_SIZE;

    // Try to reuse cached pinned staging buffer (saves ~28ms cudaHostAlloc)
    let (cached_staging, stale_ptr) = {
        let mut cache = STAGING_CACHE.lock().unwrap();
        if let Some(ref c) = *cache {
            if c.total_size >= total_staging {
                (Some(c.base_ptr), None)
            } else {
                // Cached buffer too small — take ptr, free outside the lock
                let ptr = c.base_ptr;
                *cache = None;
                (None, Some(ptr))
            }
        } else {
            (None, None)
        }
    };
    // Free stale pinned memory outside the lock
    if let Some(ptr) = stale_ptr {
        cuda_free_host(ptr as *mut u8);
    }

    let (gpu_allocs, staging_base_ptr, all_stages, all_streams) = std::thread::scope(|s| {
        // GPU alloc threads (one per device, parallel)
        let gpu_handles: Vec<_> = unique_devices
            .iter()
            .map(|&dev_idx| {
                s.spawn(move || -> std::result::Result<Vec<GpuAlloc>, String> {
                    set_cuda_device(dev_idx)
                        .map_err(|e| format!("cudaSetDevice({dev_idx}) failed: {e}"))?;

                    let mut allocs = Vec::new();
                    for shard_idx in 0..shard_offsets.len() {
                        if !shard_devices[shard_idx].contains(&dev_idx) {
                            continue;
                        }
                        let ranges = specs[shard_idx].dev_ranges.get(&dev_idx);
                        let ranges = match ranges {
                            Some(r) if !r.is_empty() => r,
                            _ => continue,
                        };
                        let mut buffers = Vec::with_capacity(ranges.len());
                        let mut range_starts = Vec::with_capacity(ranges.len());
                        for &(start, end) in ranges {
                            let size = end - start;
                            if size == 0 {
                                continue;
                            }
                            let buffer = cuda_alloc_buffer(size).map_err(|e| {
                                format!(
                                    "cudaMalloc failed shard {} dev {}: {e}",
                                    shard_idx, dev_idx
                                )
                            })?;
                            buffers.push(Arc::new(buffer));
                            range_starts.push(start);
                        }
                        if buffers.is_empty() {
                            continue;
                        }
                        allocs.push(GpuAlloc {
                            shard_idx,
                            device_idx: dev_idx,
                            buffers,
                            range_starts,
                        });
                    }
                    Ok(allocs)
                })
            })
            .collect();

        // Staging thread: reuse cached or allocate new + create streams
        type StagingResult =
            std::result::Result<(usize, Vec<Vec<usize>>, Vec<Vec<Vec<usize>>>), String>;
        let staging_handle = s.spawn(move || -> StagingResult {
            let base_ptr = if let Some(ptr) = cached_staging {
                ptr
            } else {
                cuda_host_alloc(total_staging)
                    .map_err(|e| format!("cudaHostAlloc({}MB) failed: {e}", total_staging >> 20))?
                    as usize
            };

            let mut all_stages: Vec<Vec<usize>> = Vec::with_capacity(n_active);
            for thread_idx in 0..n_active {
                let mut stages = Vec::with_capacity(N_STAGE);
                for slot in 0..N_STAGE {
                    let offset = (thread_idx * N_STAGE + slot) * CHUNK_SIZE;
                    stages.push(base_ptr + offset);
                }
                all_stages.push(stages);
            }

            let mut all_streams: Vec<Vec<Vec<usize>>> = Vec::with_capacity(n_active);
            for &(shard_idx, _, _) in active_shards.iter() {
                let targets = &shard_target_devices[shard_idx];

                let mut thread_streams = Vec::with_capacity(N_STAGE);
                for _ in 0..N_STAGE {
                    let mut slot_streams = Vec::with_capacity(targets.len());
                    for &dev_idx in targets {
                        set_cuda_device(dev_idx)
                            .map_err(|e| format!("cudaSetDevice failed: {e}"))?;
                        slot_streams.push(
                            cuda_stream_create()
                                .map_err(|e| format!("cudaStreamCreate failed: {e}"))?
                                as usize,
                        );
                    }
                    thread_streams.push(slot_streams);
                }
                all_streams.push(thread_streams);
            }
            Ok((base_ptr, all_stages, all_streams))
        });

        // Join GPU alloc threads
        let mut all_gpu = Vec::new();
        for handle in gpu_handles {
            let allocs = handle
                .join()
                .map_err(|_| LoaderError::FetchError("GPU alloc thread panicked".to_string()))?
                .map_err(LoaderError::FetchError)?;
            all_gpu.extend(allocs);
        }
        // Join staging thread
        let (base_ptr, stages, streams) = staging_handle
            .join()
            .map_err(|_| LoaderError::FetchError("Staging alloc thread panicked".to_string()))?
            .map_err(LoaderError::FetchError)?;

        Ok::<(Vec<GpuAlloc>, usize, Vec<Vec<usize>>, Vec<Vec<Vec<usize>>>), LoaderError>((
            all_gpu, base_ptr, stages, streams,
        ))
    })?;

    // Build shard → [alloc_index] lookup
    let mut shard_alloc_indices: Vec<Vec<usize>> = vec![Vec::new(); shard_paths.len()];
    for (i, alloc) in gpu_allocs.iter().enumerate() {
        shard_alloc_indices[alloc.shard_idx].push(i);
    }

    let allocs_ref = &gpu_allocs;
    let shard_alloc_ref = &shard_alloc_indices;
    let all_stages_ref = &all_stages;
    let all_streams_ref = &all_streams;

    std::thread::scope(|s| {
        let handles: Vec<_> = active_shards
            .iter()
            .enumerate()
            .map(|(thread_idx, &(shard_idx, path, offset))| {
                s.spawn(move || -> std::result::Result<(), String> {
                    let targets = &shard_alloc_ref[shard_idx];
                    let stages: Vec<*mut u8> = all_stages_ref[thread_idx]
                        .iter()
                        .map(|&p| p as *mut u8)
                        .collect();
                    let streams: Vec<Vec<*mut std::ffi::c_void>> = all_streams_ref[thread_idx]
                        .iter()
                        .map(|slot| slot.iter().map(|&s| s as *mut std::ffi::c_void).collect())
                        .collect();

                    // Get pre-computed device ranges for selective scatter
                    let shard_ranges = &specs[shard_idx].dev_ranges;

                    // Build per-target sorted range lists (indexed by target position)
                    let target_ranges: Vec<&Vec<(usize, usize)>> = targets
                        .iter()
                        .map(|&ai| {
                            let dev_idx = allocs_ref[ai].device_idx;
                            static EMPTY: Vec<(usize, usize)> = Vec::new();
                            shard_ranges.get(&dev_idx).unwrap_or(&EMPTY)
                        })
                        .collect();

                    // Open with O_DIRECT to bypass page cache
                    use std::os::unix::fs::OpenOptionsExt;
                    use std::os::unix::io::AsRawFd;
                    let file = std::fs::OpenOptions::new()
                        .read(true)
                        .custom_flags(libc::O_DIRECT)
                        .open(path)
                        .map_err(|e| format!("open O_DIRECT failed: {e}"))?;
                    let fd = file.as_raw_fd();
                    let file_size = file
                        .metadata()
                        .map_err(|e| format!("stat failed: {e}"))?
                        .len() as usize;
                    let data_size = file_size - offset;

                    // Align read start down to 4096 boundary
                    let aligned_start = offset & !(ALIGN - 1);
                    let padding = offset - aligned_start;
                    let virtual_size = data_size + padding;
                    let num_chunks = virtual_size.div_ceil(CHUNK_SIZE);

                    // Create io_uring ring (must be per-thread, not shareable)
                    let ring = io_uring::IoUring::new(N_STAGE as u32)
                        .map_err(|e| format!("io_uring_setup failed: {e}"))?;

                    // Register pre-allocated pinned buffers as fixed buffers
                    let iovecs: Vec<libc::iovec> = stages
                        .iter()
                        .map(|&ptr| libc::iovec {
                            iov_base: ptr as *mut libc::c_void,
                            iov_len: CHUNK_SIZE,
                        })
                        .collect();
                    unsafe { ring.submitter().register_buffers(&iovecs) }
                        .map_err(|e| format!("io_uring register_buffers failed: {e}"))?;

                    // Track which chunk each staging slot holds
                    let mut slot_chunk: [usize; N_STAGE] = [usize::MAX; N_STAGE];
                    let mut chunks_submitted = 0usize;
                    let mut chunks_completed = 0usize;

                    // Tensor completion tracking (streaming mode only).
                    // When on_ready is Some, we track which bytes have been confirmed
                    // (synced to GPU) and emit TensorReady callbacks as tensors complete.
                    struct PendingEmit {
                        end_offset: usize,
                        data_start: usize,
                        device_idx: usize,
                        buffer: Arc<Buffer>,
                    }
                    let mut pending: Vec<PendingEmit> = Vec::new();
                    let mut chunk_data_ends: Vec<usize> = Vec::new();
                    let mut chunk_confirmed: Vec<bool> = Vec::new();
                    let mut watermark_chunk: usize = 0;
                    let mut confirmed_bytes: usize = 0;
                    let mut emit_cursor: usize = 0;

                    if on_ready.is_some() {
                        // Build sorted pending-tensor list from all device ranges
                        for &ai in targets {
                            let ga = &allocs_ref[ai];
                            if let Some(ranges) = shard_ranges.get(&ga.device_idx) {
                                for (ri, &(start, end)) in ranges.iter().enumerate() {
                                    if ri < ga.buffers.len() {
                                        pending.push(PendingEmit {
                                            end_offset: end,
                                            data_start: start,
                                            device_idx: ga.device_idx,
                                            buffer: ga.buffers[ri].clone(),
                                        });
                                    }
                                }
                            }
                        }
                        pending.sort_by_key(|p| p.end_offset);

                        // Precompute the data-section byte where each chunk ends
                        chunk_data_ends = (0..num_chunks)
                            .map(|i| {
                                let end = if i == 0 {
                                    CHUNK_SIZE - padding
                                } else {
                                    (i + 1) * CHUNK_SIZE - padding
                                };
                                std::cmp::min(end, data_size)
                            })
                            .collect();
                        chunk_confirmed = vec![false; num_chunks];
                    }

                    // Helper: build a ReadFixed SQE for a given slot and chunk
                    let build_read_sqe = |slot_idx: usize, chunk_idx: usize| {
                        let file_pos = aligned_start + chunk_idx * CHUNK_SIZE;
                        let remaining = file_size.saturating_sub(file_pos);
                        let want = std::cmp::min(CHUNK_SIZE, remaining);
                        let to_request = (want + ALIGN - 1) & !(ALIGN - 1);
                        let to_request = std::cmp::min(to_request, CHUNK_SIZE);
                        io_uring::opcode::ReadFixed::new(
                            io_uring::types::Fd(fd),
                            stages[slot_idx],
                            to_request as u32,
                            slot_idx as u16,
                        )
                        .offset(file_pos as u64)
                        .build()
                        .user_data(slot_idx as u64)
                    };

                    // Submit initial batch of reads (fill all N_STAGE slots)
                    {
                        let initial = std::cmp::min(N_STAGE, num_chunks);
                        let mut sq = unsafe { ring.submission_shared() };
                        for (slot_idx, slot) in slot_chunk[..initial].iter_mut().enumerate() {
                            let sqe = build_read_sqe(slot_idx, slot_idx);
                            unsafe { sq.push(&sqe) }
                                .map_err(|_| "SQ full on initial submit".to_string())?;
                            *slot = slot_idx;
                            chunks_submitted += 1;
                        }
                    }
                    ring.submit()
                        .map_err(|e| format!("io_uring initial submit failed: {e}"))?;

                    // Event loop with deferred sync
                    let mut pending_resubmit: Vec<usize> = Vec::new();

                    while chunks_completed < num_chunks {
                        // Phase A: sync per-slot scatter, then resubmit reads
                        if !pending_resubmit.is_empty() {
                            let mut sq = unsafe { ring.submission_shared() };
                            for slot_idx in pending_resubmit.drain(..) {
                                for stream in &streams[slot_idx] {
                                    unsafe { cuda_stream_sync(*stream) }
                                        .map_err(|e| format!("cudaStreamSync failed: {e}"))?;
                                }

                                // Streaming mode: mark chunk confirmed, emit ready tensors
                                if let Some(ref cb) = on_ready {
                                    let completed_chunk = slot_chunk[slot_idx];
                                    chunk_confirmed[completed_chunk] = true;

                                    // Advance watermark through consecutive confirmed chunks
                                    while watermark_chunk < num_chunks
                                        && chunk_confirmed[watermark_chunk]
                                    {
                                        confirmed_bytes = chunk_data_ends[watermark_chunk];
                                        watermark_chunk += 1;
                                    }

                                    // Emit tensors whose data is fully on GPU
                                    while emit_cursor < pending.len()
                                        && pending[emit_cursor].end_offset <= confirmed_bytes
                                    {
                                        let pt = &pending[emit_cursor];
                                        cb(TensorReady {
                                            shard_idx,
                                            data_offset: pt.data_start,
                                            device_idx: pt.device_idx,
                                            buffer: pt.buffer.clone(),
                                        });
                                        emit_cursor += 1;
                                    }
                                }

                                if chunks_submitted < num_chunks {
                                    let sqe = build_read_sqe(slot_idx, chunks_submitted);
                                    unsafe { sq.push(&sqe) }
                                        .map_err(|_| "SQ full on resubmit".to_string())?;
                                    slot_chunk[slot_idx] = chunks_submitted;
                                    chunks_submitted += 1;
                                }
                            }
                        }

                        // Phase B: submit pending SQEs and wait for at least 1 CQE
                        ring.submit_and_wait(1)
                            .map_err(|e| format!("io_uring submit_and_wait failed: {e}"))?;

                        // Phase C: collect CQEs and scatter to GPUs (async)
                        let completed: Vec<(usize, usize, i32)> = {
                            let cq = unsafe { ring.completion_shared() };
                            cq.map(|cqe| {
                                let slot_idx = cqe.user_data() as usize;
                                let chunk_idx = slot_chunk[slot_idx];
                                (slot_idx, chunk_idx, cqe.result())
                            })
                            .collect()
                        };

                        for (slot_idx, chunk_idx, result) in completed {
                            if result < 0 {
                                return Err(format!(
                                    "io_uring read failed: {}",
                                    std::io::Error::from_raw_os_error(-result)
                                ));
                            }
                            let read_bytes = result as usize;

                            // Compute data range for this chunk
                            let skip = if chunk_idx == 0 { padding } else { 0 };
                            let useful = read_bytes.saturating_sub(skip);
                            let chunk_data_start = if chunk_idx == 0 {
                                0
                            } else {
                                chunk_idx * CHUNK_SIZE - padding
                            };
                            let to_copy =
                                std::cmp::min(useful, data_size.saturating_sub(chunk_data_start));
                            let chunk_data_end = chunk_data_start + to_copy;
                            let stage = stages[slot_idx];

                            // Selective scatter: binary search for overlapping ranges
                            if to_copy > 0 {
                                for (ti, &ai) in targets.iter().enumerate() {
                                    let ga = &allocs_ref[ai];
                                    let ranges = target_ranges[ti];
                                    // Find first range with end > chunk_data_start
                                    let start_ri =
                                        ranges.partition_point(|&(_, end)| end <= chunk_data_start);
                                    let mut ri = start_ri;
                                    while ri < ranges.len() && ranges[ri].0 < chunk_data_end {
                                        let (rstart, rend) = ranges[ri];
                                        let isect_start = std::cmp::max(chunk_data_start, rstart);
                                        let isect_end = std::cmp::min(chunk_data_end, rend);
                                        if isect_start < isect_end {
                                            let copy_len = isect_end - isect_start;
                                            let stage_off = skip + (isect_start - chunk_data_start);
                                            let local_offset = isect_start - rstart;
                                            unsafe {
                                                cuda_memcpy_htod_async(
                                                    (ga.buffers[ri].as_ptr() as *mut u8)
                                                        .add(local_offset),
                                                    stage.add(stage_off) as *const u8,
                                                    copy_len,
                                                    streams[slot_idx][ti],
                                                )
                                            }
                                            .map_err(|e| format!("cudaMemcpyAsync failed: {e}"))?;
                                        }
                                        ri += 1;
                                    }
                                }
                            }

                            chunks_completed += 1;
                            pending_resubmit.push(slot_idx);
                        }
                    }

                    // Final sync
                    for slot_streams in &streams {
                        for &stream in slot_streams {
                            unsafe { cuda_stream_sync(stream) }
                                .map_err(|e| format!("final cudaStreamSync failed: {e}"))?;
                        }
                    }

                    // Emit any remaining tensors (last batch not synced in Phase A)
                    if let Some(ref cb) = on_ready {
                        while emit_cursor < pending.len() {
                            let pt = &pending[emit_cursor];
                            cb(TensorReady {
                                shard_idx,
                                data_offset: pt.data_start,
                                device_idx: pt.device_idx,
                                buffer: pt.buffer.clone(),
                            });
                            emit_cursor += 1;
                        }
                    }

                    Ok(())
                })
            })
            .collect();

        for handle in handles {
            handle
                .join()
                .map_err(|_| LoaderError::FetchError("Thread panicked".to_string()))?
                .map_err(LoaderError::FetchError)?;
        }
        Ok::<(), LoaderError>(())
    })?;

    // Free CUDA streams
    for thread_streams in all_streams {
        for slot_streams in thread_streams {
            for stream in slot_streams {
                unsafe { cuda_stream_destroy(stream as *mut std::ffi::c_void) };
            }
        }
    }
    // Return pinned staging buffer to cache for reuse
    {
        let mut cache = STAGING_CACHE.lock().unwrap();
        *cache = Some(StagingCache {
            base_ptr: staging_base_ptr,
            total_size: total_staging,
        });
    }

    // Collect per-tensor buffers keyed by (data_section_offset, device_idx)
    let mut per_tensor_maps: Vec<HashMap<(usize, usize), Arc<Buffer>>> =
        vec![HashMap::new(); shard_paths.len()];
    for alloc in gpu_allocs {
        for (buffer, &range_start) in alloc.buffers.into_iter().zip(alloc.range_starts.iter()) {
            per_tensor_maps[alloc.shard_idx].insert((range_start, alloc.device_idx), buffer);
        }
    }

    let mut shard_default_loaders: Vec<Option<Arc<FileLoader>>> = vec![None; shard_paths.len()];
    for (shard_idx, path) in shard_paths.iter().enumerate() {
        let loader = FileLoader::with_backend(path, Device::Cpu, Backend::Mmap).map_err(|e| {
            LoaderError::FetchError(format!(
                "Failed to create loader for {}: {e}",
                path.display()
            ))
        })?;
        shard_default_loaders[shard_idx] = Some(Arc::new(loader));
    }

    Ok(shard_default_loaders
        .into_iter()
        .zip(per_tensor_maps)
        .map(|(loader_opt, per_tensor_map)| {
            let loader = loader_opt.unwrap();
            let per_tensor = if per_tensor_map.is_empty() {
                None
            } else {
                Some(per_tensor_map)
            };
            ShardLoadResult {
                loader,
                preloaded: None,
                per_device: None,
                per_tensor,
            }
        })
        .collect())
}

/// Single-device cuFile: parallel cuFileRead per shard to one GPU.
#[cfg(all(target_os = "linux", feature = "cufile", feature = "cuda"))]
fn load_single_cufile(
    shard_paths: &[PathBuf],
    shard_offsets: &[usize],
    config: &ScatterConfig,
) -> Result<Vec<ShardLoadResult>> {
    // Phase 1 (sequential): create loaders + allocate + register buffers
    let mut loaders_and_buffers: Vec<(Arc<FileLoader>, Buffer)> = Vec::with_capacity(shard_paths.len());
    for (path, &offset) in shard_paths.iter().zip(shard_offsets.iter()) {
        let loader = FileLoader::with_backend(path, config.device, config.backend).map_err(|e| {
            LoaderError::FetchError(format!(
                "Failed to create loader for {}: {e}",
                path.display()
            ))
        })?;
        let buffer = loader
            .alloc_buffer(offset, loader.file_size())
            .map_err(|e| {
                LoaderError::FetchError(format!(
                    "Failed to allocate buffer for {}: {e}",
                    path.display()
                ))
            })?;
        super::cufile_buf_register(buffer.as_ptr(), buffer.len());
        loaders_and_buffers.push((Arc::new(loader), buffer));
    }

    // Phase 2 (parallel): only cuFileRead — no allocation or registration
    std::thread::scope(|s| {
        let handles: Vec<_> = loaders_and_buffers
            .iter()
            .zip(shard_offsets.iter())
            .map(|((loader, buffer), &offset)| {
                s.spawn(move || -> std::result::Result<(), String> {
                    loader
                        .read_into(buffer, offset)
                        .map_err(|e| format!("cuFileRead failed: {e}"))?;
                    Ok(())
                })
            })
            .collect();
        for handle in handles {
            handle
                .join()
                .map_err(|_| LoaderError::FetchError("Thread panicked".to_string()))?
                .map_err(LoaderError::FetchError)?;
        }
        Ok::<(), LoaderError>(())
    })?;

    // Phase 3 (sequential): deregister and collect
    Ok(loaders_and_buffers
        .into_iter()
        .map(|(loader, buffer)| {
            super::cufile_buf_deregister(buffer.as_ptr());
            ShardLoadResult {
                loader,
                preloaded: Some(Arc::new(buffer)),
                per_device: None,
                per_tensor: None,
            }
        })
        .collect())
}

/// Default path: create loaders + preload concurrently.
fn load_default(
    shard_paths: &[PathBuf],
    shard_offsets: &[usize],
    config: &ScatterConfig,
) -> Result<Vec<ShardLoadResult>> {
    #[cfg(feature = "cuda")]
    let is_cuda = matches!(config.device, Device::Cuda(_));
    #[cfg(not(feature = "cuda"))]
    let is_cuda = false;

    std::thread::scope(|s| {
        let handles: Vec<_> = shard_paths
            .iter()
            .zip(shard_offsets.iter())
            .map(|(path, &offset)| {
                s.spawn(move || -> std::result::Result<ShardLoadResult, String> {
                    let loader = FileLoader::with_backend(path, config.device, config.backend)
                        .map_err(|e| {
                            format!("Failed to create loader for {}: {e}", path.display())
                        })?;

                    let preloaded = if is_cuda && config.is_single {
                        loader.fetch(offset, loader.file_size()).ok().map(Arc::new)
                    } else {
                        None
                    };

                    Ok(ShardLoadResult {
                        loader: Arc::new(loader),
                        preloaded,
                        per_device: None,
                        per_tensor: None,
                    })
                })
            })
            .collect();

        let mut results = Vec::with_capacity(handles.len());
        for handle in handles {
            results.push(
                handle
                    .join()
                    .map_err(|_| LoaderError::FetchError("Thread panicked".to_string()))?
                    .map_err(LoaderError::FetchError)?,
            );
        }
        Ok(results)
    })
}
