# Plan: hmll io_uring Integration for Fast Multi-Device Model Loading

## Performance Targets (fio baselines, io_uring, 4 jobs, iodepth=64, 1M bs)

| Filesystem | Throughput | IOPS | Avg Latency |
|---|---|---|---|
| `/tmp` (local NVMe) | **1,020 MiB/s** (1.07 GB/s) | 1,019 | 251 ms |
| `/fsx/luc` (Lustre) | **9,451 MiB/s** (9.91 GB/s) | 9,450 | 27 ms |

These are the hardware ceilings — we aim to approach them end-to-end through `safe_open`.

---

## Architecture

```
Python  safe_open(files, framework, device_map=... | shard_plan=...)
          │
          ├─ legacy path: mmap → framework .to(device)     [unchanged]
          │
          └─ hmll path:   TensorLoader → WeightLoader(io_uring)
                          ├── fetchv: batch multiple tensors in one syscall
                          ├── device_map mode: whole tensors → per-device loaders
                          └── shard_plan mode: contiguous shard slices → rank GPU
```

`device_map` and `shard_plan` are mutually exclusive args. Both use hmll underneath.
Everything gated behind `#[cfg(feature = "hmll")]` + experimental warnings.

---

## Task Breakdown

### Task 0.5: hmll — Add `take_context_error` helper to Rust `WeightLoader`

**Why**: The C API uses sticky context errors (`ctx->error`). Once set, the context is poisoned — every function guards with `if (hmll_check(ctx->error)) return -1`. The C code never clears it; that's the caller's job. The Rust side currently does a manual 5-line extract-reset-convert dance in `fetch()` that would be duplicated in `fetchv()`. A private helper deduplicates this. No C header change needed — `static inline` functions aren't available via FFI anyway (bindgen can't bind them), so the Rust side just operates on the struct fields directly.

**Where**: `~/hmll/lib/rust/hmll/src/loader.rs`

**What**:
```rust
// loader.rs — private helper, replaces the manual 5-line dance in fetch() and fetchv()
fn take_context_error(&mut self) -> Error {
    let err = self.context.error;
    self.context.error = hmll_sys::hmll_error {
        code: hmll_sys::HMLL_ERR_SUCCESS,
        sys_err: 0,
    };
    Error::from_hmll_error(err)
}
```
- Refactor existing `fetch()` error handling (lines 260-267) to use `self.take_context_error()`
- `fetchv()` (Task 1) will use it too

---

### Task 1: hmll — Expose `fetchv` in Rust bindings

**Why**: `fetchv` is the single biggest perf lever. The C implementation (`iouring.c:278-464`) does round-robin SQE packing across N buffers with CCA-tuned batching. Currently only exposed at the C layer — the Rust `WeightLoader` only has `fetch()` (one buffer at a time). Without this, loading 100 tensors means 100 sequential io_uring submission cycles instead of one batched operation.

**Where**: `~/hmll/lib/rust/hmll/src/loader.rs`

**What**:
- Add `WeightLoader::fetchv()` method wrapping `hmll_sys::hmll_fetchv`
- Signature: `fn fetchv(&mut self, ranges: &[Range], file_index: usize) -> Result<Vec<Buffer>>`
- Allocate N `hmll_iobuf` buffers, build offsets array, call `hmll_fetchv`, wrap results as `Vec<Buffer>`
- Tests: multi-buffer correctness, empty ranges, mixed sizes
- See `task1_fetchv_plan.md` for detailed implementation plan

---

### Task 2: hmll — Add `FromStr` for `Device` and `Display` improvements

**Why**: Ergonomic device parsing for integration. Currently `Device` is just `{Cpu, Cuda}` with no `FromStr`, no index support, and Display gives "CPU"/"CUDA" instead of the standard "cpu"/"cuda:0" format. The safetensors `device_map` uses strings like `"cuda:0"`, `"cpu"` — we need clean conversions.

**Where**: `~/hmll/lib/rust/hmll/src/device.rs`

**What**:
- `FromStr` impl: parse `"cpu"` → `Cpu`, `"cuda"` → `Cuda`, `"cuda:N"` → `Cuda` (index ignored at hmll level, it's set via CUDA_VISIBLE_DEVICES / cudaSetDevice by the caller)
- Update `Display` to lowercase: `"cpu"`, `"cuda"` (matches convention in PyTorch/safetensors)
- Tests for all parse cases + error handling

---

### Task 3: hmll — Add `fetchv` to mmap backend (if missing)

**Why**: The mmap backend needs `fetchv` support too for the non-Linux fallback path and for CPU-only loading. The C mmap backend already has `hmll_mmap_fetchv_range_impl` — verify it's wired up and the Rust wrapper can call it transparently.

**Where**: `~/hmll/lib/unix/backend/mmap.c`, verify via `~/hmll/lib/rust/hmll/src/loader.rs`

**What**:
- Verify `fetchv_range_impl_` function pointer is set in mmap init (check `~/hmll/lib/unix/loader.c`)
- The Rust `fetchv()` from Task 1 should work for both backends transparently since it calls through the C vtable
- Add mmap-specific test for `fetchv`

---

### Task 4: safetensors — Core `TensorLoader` in `loader.rs`

**Why**: This is the bridge between safetensors metadata and hmll I/O. It resolves tensor names to (file_index, byte_range), handles multi-shard models, and provides the tensor-level fetch API that the Python bindings will call.

**Where**: `safetensors/src/loader.rs` (currently a stub)

**What**:
```rust
pub struct TensorLoader {
    sources: Vec<hmll::Source>,
    loader: hmll::WeightLoader<'static>, // 'static via self-referential Arc trick
    shard_meta: Vec<ShardMeta>,          // per-shard: header offset + Metadata
    tensor_index: HashMap<String, (usize, TensorInfo)>, // name → (shard_idx, info)
}

struct ShardMeta {
    offset: usize,      // n + 8 (header_len + header_size_bytes)
    metadata: Metadata,
}
```

Methods:
- `open(paths, device) -> Result<Self>` — open all shards, read + parse headers via hmll `fetch`, build tensor index
- `open_index(index_path, device) -> Result<Self>` — parse `index.json`, resolve shard paths, delegate to `open()`
- `fetch_tensor(name) -> Result<(Buffer, &TensorInfo)>` — single tensor fetch
- `fetch_tensors(names) -> Result<Vec<(Buffer, &TensorInfo)>>` — batch fetch via `fetchv` (perf critical)
- `keys()`, `metadata()`, `num_shards()`, accessor methods

For `device_map` mode, we need one `WeightLoader` per distinct `hmll::Device`. The TensorLoader can hold a `HashMap<hmll::Device, WeightLoader>` or just a CPU + optional CUDA loader pair (models typically only target 2 devices max).

---

### Task 5: safetensors — Wire `shard_plan` into `TensorLoader`

**Why**: Tensor-parallel loading needs shard-aware fetching. The `ShardPlan` already computes `ShardSlice` (contiguous byte ranges or narrow-after-load descriptors). The TensorLoader needs to apply these to compute the correct byte ranges for hmll.

**Where**: `safetensors/src/loader.rs`

**What**:
- `TensorLoader` optionally holds a `ShardPlan` + rank + world_size
- `fetch_tensor(name)` auto-applies sharding when plan is set:
  - `ShardSlice::Contiguous { start, end }` → fetch `[shard_offset + start, shard_offset + end)` only
  - `ShardSlice::NarrowAfterLoad { shape, dim, start, len }` → fetch full tensor, return slice metadata alongside buffer
  - `ShardSlice::FullCopy` → fetch full tensor
- Return type should carry the slice info so Python can narrow if needed
- `fetch_tensors()` batch version should apply sharding per-tensor and batch the fetches

---

### Task 6: safetensors — `DeviceMapLoader` for multi-device loading

**Why**: `device_map` mode requires loading different tensors to different devices within a single process. This needs a separate orchestration layer on top of `TensorLoader`.

**Where**: `safetensors/src/device_map.rs` (behind `#[cfg(feature = "hmll")]`)

**What**:
- `DeviceMapLoader` struct that:
  - Takes a `DeviceMap` (tensor name → device)
  - Extracts unique `hmll::Device`s from the map
  - Creates one `TensorLoader` per unique device
  - Routes `fetch_tensor(name)` to the correct loader based on the map
- `impl From<&device_map::Device> for hmll::Device` for conversions
- This is a **separate task** from shard_plan integration (Task 7 focuses on shard_plan only)

**Deferred**: Implement after Task 7 (shard_plan) is working end-to-end.

---

### Task 7: Python bindings — `Storage::Hmll` + extended `safe_open` (shard_plan only)

**Why**: This is the user-facing integration point. `safe_open` needs new parameters for shard_plan and multi-file loading.

**Scope**: This task focuses on `shard_plan` mode only. `device_map` support will be added in Task 6 after this is working.

**Where**: `bindings/python/src/lib.rs`, `bindings/python/Cargo.toml`

**What**:

Cargo.toml additions:
```toml
hmll = { path = "../../hmll/lib/rust/hmll", optional = true }
[features]
hmll = ["dep:hmll", "safetensors/hmll"]
```

New `Storage::Hmll` variant holding a single `TensorLoader` (one device).

Extended `safe_open` signature:
```python
safe_open(
    filename,           # str | Path | list[str | Path]
    framework,          # str
    device=None,        # str | int | None
    shard_plan=None,    # dict[str, str] | None
    rank=None,          # int | None
    world_size=None,    # int | None
)
```

Validation:
- `shard_plan` requires `rank` + `world_size` → `ValueError`
- `shard_plan` requires CUDA device → `ValueError` if CPU

Experimental warning on construction:
```python
warnings.warn(
    "safetensors: Using experimental hmll/io_uring backend. "
    "This may cause crashes. Use at your own risk.",
    UserWarning, stacklevel=2
)
```

Tensor retrieval:
1. `get_tensor(name)` → `TensorLoader::fetch_tensor()` → `buffer.as_slice()` → framework tensor
2. If `NarrowAfterLoad`, apply `.narrow(dim, start, len).contiguous()` on the framework tensor

---

### Task 8: Python bindings — Build system & runtime feature detection

**Why**: hmll must be opt-in at build time (experimental C dep), and Python code needs to detect availability at runtime.

**Where**: `bindings/python/build.rs`, `bindings/python/pyproject.toml`, `bindings/python/src/lib.rs`

**What**:
- Build-time: `SAFETENSORS_ENABLE_HMLL=1` env var → `build.rs` enables `hmll` Cargo feature
- Runtime: expose `safetensors.has_hmll: bool` in Python module
- When hmll features used but not available → clear error: "Built without hmll support. Rebuild with SAFETENSORS_ENABLE_HMLL=1 pip install ."

**Wheel distribution strategy** (future):
- Currently: users build from source with `SAFETENSORS_ENABLE_HMLL=1 pip install . --no-binary safetensors`
- Option A: Publish two wheel sets (`safetensors` and `safetensors-hmll` packages)
- Option B: Single package, `[hmll]` extra pulls in `safetensors-hmll` as dependency
- Defer decision until feature is stable; env var approach works for early adopters

---

### Task 9: Testing

**Where**: `safetensors/src/loader.rs` (Rust), `bindings/python/tests/` (Python)

**What**:

Rust:
- TensorLoader open/fetch with real safetensors files (create test fixtures)
- Multi-shard open + fetch_tensors batch correctness
- ShardPlan integration (contiguous + narrow-after-load + replicate)
- device_map conversion correctness
- Error paths (missing tensor, bad file, empty sources)

Python:
- `safe_open` with shard_plan — shard slices match reference
- Multi-file list and index.json modes
- shard_plan validation (requires rank + world_size)
- Warning emission
- `has_hmll` runtime check
- (device_map tests deferred to Task 6)

---

### Task 10: Benchmarks & Demo

**Why**: Blog post / tweet material. Need compelling before/after numbers.

**Where**: New benchmark scripts, can be Python or Rust

**What**:
- Rust criterion: `hmll fetch` vs `mmap read` for 100MB, 1GB, 10GB tensors
- Python e2e: `safe_open` mmap path vs hmll path, on `/tmp` and `/fsx/luc`
- Multi-shard: 8-shard model load time
- Target metrics: total throughput (GB/s), time-to-first-tensor, time-to-all-tensors
- Compare against fio baselines from above

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| fetchv vs sequential fetch | Expose `fetchv` for batch tensor loading | Round-robin SQE packing + CCA = much higher throughput than sequential |
| One loader per device | CPU + optional CUDA loader | hmll binds device at init; models target 1-2 devices |
| Backend selection | Force `IoUring` on Linux, `Mmap` elsewhere | io_uring is the point; mmap fallback for portability |
| Multi-file input | Accept list, dir, or index.json | Covers all HF model distribution patterns |
| Shard narrowing | Framework-side `.narrow()` for rowwise | Non-contiguous; cheaper to narrow on GPU post-load |
| Feature gating | Build-time env var + Cargo feature | Prevents accidental inclusion of experimental C dep |
| Device index | Handled by caller (CUDA_VISIBLE_DEVICES), not hmll | hmll Device is {Cpu, Cuda} — index is a runtime/env concern |
| hmll Display format | Lowercase "cpu"/"cuda" | Match PyTorch/safetensors convention |

---

## Post-Integration: Ergonomics Pass

After core integration is complete, revisit hmll Rust bindings for API ergonomics:

### 1. `&self` fetch API via interior mutability

The C API stores errors in `ctx->error`, requiring `&mut self` for `fetch()`/`fetchv()`. This is awkward for callers. Options:

- **`UnsafeCell`**: Zero overhead, but requires documenting thread-safety constraints (no concurrent calls to same loader)
- **`RefCell`**: Runtime borrow checking, small overhead per call

Recommendation: `UnsafeCell` with documented constraints, since loaders are typically single-threaded.

### 2. Audit `Send`/`Sync` implementations

Current `unsafe impl Send + Sync` on `WeightLoader` and `Source` may not be valid if the underlying C context isn't thread-safe. Audit:

- Is `hmll_t` safe to send between threads?
- Are concurrent reads from different threads safe?
- Document actual thread-safety guarantees

### 3. Document thread-safety constraints

Add crate-level documentation clarifying:

- Single-threaded use recommended
- Concurrent `fetch()` calls on same loader are UB (if using `UnsafeCell`)
- Multiple loaders in different threads are fine (each has own context)

---

## Future Enhancements

### shard_plan support for mmap backend

The sharding logic (computing byte ranges, narrowing after load) is independent of the I/O backend. The mmap backend could also benefit from shard_plan:

- Read just `&mmap[start..end]` for contiguous slices instead of full tensor
- Apply `.narrow()` on framework tensor for rowwise sharding
- Same `shard_plan`, `rank`, `world_size` parameters work for both backends
- `ShardConfig` and validation logic are already backend-agnostic

This enables tensor-parallel loading without requiring hmll, useful for:
- Non-Linux platforms (no io_uring)
- CPU-only deployments
- Simpler setup (no hmll C dependency)
