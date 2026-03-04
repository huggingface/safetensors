# Task 7: Python Bindings - Standalone `safe_open_hmll`

## Overview

Add a standalone `safe_open_hmll` class for hmll/io_uring-based tensor loading with sharding. Completely separate from existing `safe_open` - no integration, no routing, no changes to existing code.

---

## Design

- New `safe_open_hmll` PyClass in `bindings/python/src/hmll.rs`
- Feature-gated with `#[cfg(feature = "hmll")]`
- Users explicitly import and use it
- Initial scope: PyTorch only, CUDA only, shard_plan required

```python
from safetensors import safe_open_hmll

with safe_open_hmll(
    "model.safetensors",
    framework="pt",
    device="cuda:0",
    shard_plan={"*.weight": "colwise"},
    rank=0,
    world_size=2,
) as f:
    tensor = f.get_tensor("layer.0.weight")
```

---

## File Structure

```
bindings/python/src/
├── lib.rs          # add: mod hmll, register class, has_hmll
├── hmll.rs         # NEW: safe_open_hmll implementation
└── view.rs         # unchanged
```

---

## Implementation

### `hmll.rs`

```rust
//! Standalone hmll/io_uring backend for tensor-parallel loading.
//!
//! Supports: PyTorch, CUDA, shard_plan

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::PyByteArray;

use safetensors::tensor::Dtype;
use safetensors::loader::{TensorLoader, FetchResult};
use safetensors::shard_plan::{ShardPlan, ShardSlice, ShardStrategy};

use crate::{SafetensorError, Device, Framework};

/// Validated shard configuration
struct ShardConfig {
    plan: ShardPlan,
    rank: usize,
}

fn validate_and_parse(
    device: &Option<Device>,
    shard_plan: Option<HashMap<String, String>>,
    rank: Option<usize>,
    world_size: Option<usize>,
) -> PyResult<ShardConfig> {
    let (raw_plan, rank, world_size) = match (shard_plan, rank, world_size) {
        (Some(p), Some(r), Some(ws)) => (p, r, ws),
        (Some(_), _, _) => {
            return Err(SafetensorError::new_err(
                "shard_plan requires both rank and world_size"
            ))
        }
        (None, None, None) => {
            return Err(SafetensorError::new_err(
                "safe_open_hmll requires shard_plan, rank, and world_size"
            ))
        }
        _ => {
            return Err(SafetensorError::new_err(
                "rank and world_size require shard_plan"
            ))
        }
    };

    if !matches!(device, Some(Device::Cuda(_))) {
        return Err(SafetensorError::new_err(
            "safe_open_hmll requires a CUDA device"
        ));
    }

    let patterns = raw_plan
        .into_iter()
        .map(|(pattern, strategy_str)| {
            strategy_str
                .parse::<ShardStrategy>()
                .map(|s| (pattern, s))
                .map_err(|e| SafetensorError::new_err(e.to_string()))
        })
        .collect::<PyResult<HashMap<_, _>>>()?;

    Ok(ShardConfig {
        plan: ShardPlan::new(patterns, world_size),
        rank,
    })
}

struct OpenHmll {
    loader: RefCell<TensorLoader>,
    device: Device,
}

impl OpenHmll {
    fn new(filename: PathBuf, device: Device, config: ShardConfig) -> PyResult<Self> {
        // Experimental warning
        Python::with_gil(|py| -> PyResult<()> {
            let warnings = py.import("warnings")?;
            warnings.call_method1(
                "warn",
                (
                    "safetensors: Using experimental hmll/io_uring backend.",
                    py.get_type::<pyo3::exceptions::PyUserWarning>(),
                ),
            )?;
            Ok(())
        })?;

        let hmll_device = match &device {
            Device::Cuda(idx) => hmll::Device::Cuda(*idx as u8),
            _ => unreachable!(),
        };

        let loader = TensorLoader::open(&[&filename], hmll_device)
            .map_err(|e| SafetensorError::new_err(e.to_string()))?
            .with_shard_plan(config.plan, config.rank);

        Ok(Self {
            loader: RefCell::new(loader),
            device,
        })
    }

    fn keys(&self) -> Vec<String> {
        self.loader.borrow().keys().cloned().collect()
    }

    fn get_tensor(&self, name: &str) -> PyResult<PyObject> {
        let result = self.loader.borrow_mut()
            .fetch_tensor(name)
            .map_err(|e| SafetensorError::new_err(e.to_string()))?;

        Python::with_gil(|py| {
            let torch = py.import("torch")?;

            let data = result.buffer.as_slice()
                .map_err(|e| SafetensorError::new_err(e.to_string()))?;

            // Shape from sharding
            let shape: Vec<usize> = match &result.shard_slice {
                Some(ShardSlice::Contiguous { shape, .. }) => shape.clone(),
                _ => result.info.shape.clone(),
            };

            // Create tensor
            let dtype = self.torch_dtype(torch, result.info.dtype)?;
            let array = PyByteArray::new(py, data);
            let tensor = torch
                .call_method1("frombuffer", (array, torch.getattr("uint8")?))?
                .call_method1("view", (dtype,))?
                .call_method1("reshape", (shape,))?;

            // Move to CUDA
            let device_str = match &self.device {
                Device::Cuda(idx) => format!("cuda:{}", idx),
                _ => unreachable!(),
            };
            let tensor = tensor.call_method1("to", (device_str,))?;

            // Narrow if needed
            let tensor = match &result.shard_slice {
                Some(ShardSlice::NarrowAfterLoad { dim, start, len, .. }) => {
                    tensor
                        .call_method1("narrow", (*dim as i64, *start as i64, *len as i64))?
                        .call_method0("contiguous")?
                }
                _ => tensor,
            };

            Ok(tensor.into())
        })
    }

    fn torch_dtype(&self, torch: &Bound<'_, PyModule>, dtype: Dtype) -> PyResult<PyObject> {
        let name = match dtype {
            Dtype::F32 => "float32",
            Dtype::F16 => "float16",
            Dtype::BF16 => "bfloat16",
            Dtype::F64 => "float64",
            Dtype::I64 => "int64",
            Dtype::I32 => "int32",
            Dtype::I16 => "int16",
            Dtype::I8 => "int8",
            Dtype::U8 => "uint8",
            other => return Err(SafetensorError::new_err(format!("Unsupported dtype: {:?}", other))),
        };
        Ok(torch.getattr(name)?.into())
    }
}

#[pyclass]
#[allow(non_camel_case_types)]
pub struct safe_open_hmll {
    inner: Option<OpenHmll>,
}

impl safe_open_hmll {
    fn inner(&self) -> PyResult<&OpenHmll> {
        self.inner.as_ref().ok_or_else(|| SafetensorError::new_err("File is closed"))
    }
}

#[pymethods]
impl safe_open_hmll {
    #[new]
    #[pyo3(signature = (filename, framework, device, shard_plan, rank, world_size))]
    pub fn new(
        filename: PathBuf,
        framework: Framework,
        device: Option<Device>,
        shard_plan: Option<HashMap<String, String>>,
        rank: Option<usize>,
        world_size: Option<usize>,
    ) -> PyResult<Self> {
        if framework != Framework::Pytorch {
            return Err(SafetensorError::new_err(
                "safe_open_hmll only supports PyTorch (framework='pt')"
            ));
        }

        let config = validate_and_parse(&device, shard_plan, rank, world_size)?;
        let inner = OpenHmll::new(filename, device.unwrap(), config)?;

        Ok(Self { inner: Some(inner) })
    }

    pub fn keys(&self) -> PyResult<Vec<String>> {
        Ok(self.inner()?.keys())
    }

    pub fn get_tensor(&self, name: &str) -> PyResult<PyObject> {
        self.inner()?.get_tensor(name)
    }

    pub fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    pub fn __exit__(&mut self, _exc_type: PyObject, _exc_value: PyObject, _traceback: PyObject) {
        self.inner = None;
    }
}
```

### `lib.rs` Changes

```rust
// At top
#[cfg(feature = "hmll")]
mod hmll;

// In _safetensors_rust module registration
#[pymodule]
fn _safetensors_rust(m: &PyBound<'_, PyModule>) -> PyResult<()> {
    // ... existing ...

    #[cfg(feature = "hmll")]
    m.add_class::<hmll::safe_open_hmll>()?;

    #[cfg(feature = "hmll")]
    m.add("has_hmll", true)?;
    #[cfg(not(feature = "hmll"))]
    m.add("has_hmll", false)?;

    Ok(())
}
```

### `Cargo.toml` Changes

```toml
[features]
default = []
hmll = ["safetensors/hmll", "dep:hmll"]

[dependencies]
hmll = { path = "../../../hmll/lib/rust/hmll", optional = true }

[dependencies.safetensors]
path = "../../safetensors"
```

---

## File Changes Summary

| File | Change |
|------|--------|
| `src/hmll.rs` | NEW |
| `src/lib.rs` | +5 lines |
| `Cargo.toml` | +4 lines |
| `build.rs` | NEW (env var detection) |
| `pyproject.toml` | +1 line (hmll extra) |

---

## Usage

```python
from safetensors import safe_open_hmll

shard_plan = {
    "*.q_proj.weight": "colwise",
    "*.o_proj.weight": "rowwise",
}

with safe_open_hmll(
    "model.safetensors",
    framework="pt",
    device="cuda:0",
    shard_plan=shard_plan,
    rank=0,
    world_size=2,
) as f:
    print(f.keys())
    tensor = f.get_tensor("layer.0.q_proj.weight")
```

---

## Supported Features

| Feature | Status |
|---------|--------|
| PyTorch | ✅ |
| Other frameworks | ❌ |
| CUDA | ✅ |
| CPU | ❌ |
| shard_plan | ✅ required |
| get_tensor | ✅ |
| get_slice | ❌ |
| keys | ✅ |
| Multi-file | ❌ future |

---

## Testing

```python
def test_safe_open_hmll_basic():
    """Load tensor with colwise sharding"""

def test_safe_open_hmll_rowwise():
    """Load tensor with rowwise sharding (narrow)"""

def test_safe_open_hmll_requires_cuda():
    """Error on CPU device"""

def test_safe_open_hmll_requires_pytorch():
    """Error on non-pytorch framework"""

def test_safe_open_hmll_requires_shard_plan():
    """Error without shard_plan"""

def test_has_hmll():
    """Runtime feature detection"""
```
