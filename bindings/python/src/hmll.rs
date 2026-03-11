//! Self-contained hmll implementation to separate and feature flag easily while it's currently
//! under development and testing.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;

use pyo3::prelude::*;

use pyo3::types::PyList;
use safetensors::loader::TensorLoader;
use safetensors::shard_plan::{ShardPatternConfig, ShardPlan, ShardSlice, ShardStrategy};

use crate::dlpack;

use crate::{Device, Framework, SafetensorError};

struct ShardConfig {
    plan: ShardPlan,
    rank: usize,
}

#[derive(FromPyObject)]
#[pyo3(from_item_all)]
struct PyShardPatternConfig {
    strategy: String,
    #[pyo3(default)]
    ranks: Option<Vec<usize>>,
}

fn validate_shard_config(
    device: &Device,
    raw_plan: HashMap<String, PyShardPatternConfig>,
    rank: usize,
    world_size: usize,
) -> PyResult<(hmll::Device, ShardConfig)> {
    let idx = if let Device::Cuda(idx) = device {
        idx
    } else {
        return Err(SafetensorError::new_err(
            "sharding requires device to be CUDA, shard_plan is not None but device is CPU",
        ));
    };

    let patterns = raw_plan
        .into_iter()
        .map(|(pattern, config)| {
            let strategy = config.strategy.parse::<ShardStrategy>().map_err(|e| {
                SafetensorError::new_err(format!("error parsing shard strategy: {e}"))
            })?;
            Ok((
                pattern,
                ShardPatternConfig {
                    strategy,
                    ranks: config.ranks,
                },
            ))
        })
        .collect::<PyResult<HashMap<_, _>>>()?;
    Ok((
        hmll::Device::Cuda((*idx) as u8),
        ShardConfig {
            plan: ShardPlan::new(patterns, world_size),
            rank,
        },
    ))
}

struct OpenHmll {
    loader: RefCell<TensorLoader>,
    device: hmll::Device,
}

impl OpenHmll {
    pub fn new(
        filename: FilenameArg,
        device: hmll::Device,
        shard_config: ShardConfig,
    ) -> PyResult<Self> {
        eprintln!("warning: Using experimental hmll/io_uring backend, use at your own risk.");
        let loader = match filename {
            FilenameArg::Index(path) => TensorLoader::open_index(path, device)
                .map_err(|e| SafetensorError::new_err(format!("error creating loader: {e}")))?,
            FilenameArg::Single(path) => TensorLoader::open(&[path], device)
                .map_err(|e| SafetensorError::new_err(format!("error creating loader: {e}")))?,
            FilenameArg::Multiple(paths) => TensorLoader::open(&paths, device)
                .map_err(|e| SafetensorError::new_err(format!("error creating loader: {e}")))?,
        };

        let loader = RefCell::new(
            loader
                .with_shard_plan(shard_config.plan, shard_config.rank)
                .map_err(|e| {
                    SafetensorError::new_err(format!("error applying shard plan to loader: {e}"))
                })?,
        );

        Ok(Self { loader, device })
    }

    pub fn keys(&self) -> Vec<String> {
        self.loader.borrow().keys().cloned().collect()
    }

    pub fn load_tensors(&self, py: Python<'_>) -> PyResult<Vec<(String, PyObject)>> {
        let mut loader = self.loader.borrow_mut();
        let results = loader
            .fetch_all_tensors()
            .map_err(|e| SafetensorError::new_err(format!("error loading tensors: {e}")))?;
        let torch = py.import("torch")?;

        results
            .into_iter()
            .map(|(name, result)| {
                let shape = result
                    .shard_slice
                    .as_ref()
                    .and_then(ShardSlice::shape)
                    .expect("shape should not be None")
                    .to_vec();
                let capsule = dlpack::buffer_to_capsule(
                    py,
                    result.buffer,
                    shape,
                    result.info.dtype,
                    self.device,
                )?;
                let tensor = torch.call_method1("from_dlpack", (capsule,))?;

                Ok((name, tensor.into()))
            })
            .collect()
    }
}

enum FilenameArg {
    Index(PathBuf),
    Single(PathBuf),
    Multiple(Vec<PathBuf>),
}

impl TryFrom<PathBuf> for FilenameArg {
    type Error = PyErr;

    fn try_from(path: PathBuf) -> Result<Self, Self::Error> {
        if path.is_dir() {
            for entry in std::fs::read_dir(&path)
                .map_err(|e| SafetensorError::new_err(format!("error finding .index.json: {e}")))?
            {
                let entry = entry.map_err(|e| {
                    SafetensorError::new_err(format!("error finding .index.json: {e}"))
                })?;
                let entry_path = entry.path();
                if entry_path.to_string_lossy().ends_with(".index.json") {
                    return Ok(FilenameArg::Index(entry_path));
                }
            }
            Err(SafetensorError::new_err(format!(
                ".index.json file was not found in provided directory: {}",
                path.to_string_lossy()
            )))
        } else if path.to_string_lossy().ends_with(".index.json") {
            Ok(FilenameArg::Index(path))
        } else {
            Ok(FilenameArg::Single(path))
        }
    }
}

impl<'py> FromPyObject<'py> for FilenameArg {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(list) = ob.downcast::<PyList>() {
            let paths = list.extract()?;
            return Ok(FilenameArg::Multiple(paths));
        }
        let path: PathBuf = ob.extract()?;
        FilenameArg::try_from(path)
    }
}

#[pyclass(unsendable)]
#[allow(non_camel_case_types)]
pub struct safe_open_hmll {
    inner: Option<OpenHmll>,
}

impl safe_open_hmll {
    fn inner(&self) -> PyResult<&OpenHmll> {
        self.inner
            .as_ref()
            .ok_or_else(|| SafetensorError::new_err("File is closed"))
    }
}

#[pymethods]
impl safe_open_hmll {
    #[new]
    #[pyo3(signature = (filename, framework, device, shard_plan, rank, world_size))]
    fn new(
        filename: FilenameArg,
        framework: Framework,
        device: Device,
        shard_plan: HashMap<String, PyShardPatternConfig>,
        rank: usize,
        world_size: usize,
    ) -> PyResult<Self> {
        if framework != Framework::Pytorch {
            return Err(SafetensorError::new_err(
                "safe_open_hmll only supports pytorch (framework='pt')",
            ));
        }

        let (hmll_device, config) = validate_shard_config(&device, shard_plan, rank, world_size)?;
        let inner = OpenHmll::new(filename, hmll_device, config)?;

        Ok(Self { inner: Some(inner) })
    }

    pub fn keys(&self) -> PyResult<Vec<String>> {
        Ok(self.inner()?.keys())
    }

    pub fn load_tensors(&self, py: Python<'_>) -> PyResult<Vec<(String, PyObject)>> {
        self.inner()?.load_tensors(py)
    }

    pub fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    pub fn __exit__(
        &mut self,
        _exc_type: Option<PyObject>,
        _exc_value: Option<PyObject>,
        _traceback: Option<PyObject>,
    ) {
        self.inner = None;
    }
}
