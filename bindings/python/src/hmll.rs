//! Self-contained hmll implementation to separate and feature flag easily while it's currently
//! under developmemt and testing

use safetensors::loader::TensorLoader;
use safetensors::shard_plan::{ShardPlan, ShardStrategy};
use std::cell::RefCell;

struct ShardConfig {
    plan: ShardPlan,
    rank: usize,
}

pub struct OpenHmll {
    shard_config: Option<ShardConfig>,
}

impl OpenHmll {
    #[pyo3(signature = (f, framework, device=Some(Device::Cpu), shard_plan=None, rank=None, world_size=None))]
    pub fn new(
        f: PyObject,
        framework: Framework,
        device: Option<Device>,
        shard_plan: Option<HashMap<String, String>>,
        rank: Option<usize>,
        world_size: Option<usize>,
    ) -> Self {
        if filename.is_dir() {
            // TODO: 1. find `index.json` file
            // 2. collect all safetensors shards
        } else if filename.ends_with("index.json") {
        }

        let filename = Python::with_gil(|py| -> PyResult<PathBuf> {
            let _ = f.getattr(py, "fileno")?;
            let filename = f.getattr(py, "name")?;
            let filename: PathBuf = filename.extract(py)?;
            Ok(filename)
        })?;
        let inner = Some(Open::new(filename, framework, device)?);
        Ok(Self { inner })
    }
}

fn validate_shard_config(
    device: Option<&Device>,
    shard_plan: Option<HashMap<String, String>>,
    rank: Option<usize>,
    world_size: Option<usize>,
) -> PyResult<Option<ShardConfig>> {
    Ok(match (shard_plan, rank, world_size) {
        (None, None, None) => None,
        (Some(raw_plan), Some(rank), Some(world_size)) => {
            if !matches!(device, Some(Device::Cuda(_))) {
                return Err(SafetensorError::new_err(
                    "sharding requires device to be CUDA, shard_plan is not None but device is CPU",
                ));
            }
            let patterns = raw_plan
                .into_iter()
                .map(|(pattern, strategy_str)| {
                    strategy_str
                        .parse::<ShardStrategy>()
                        .map(|strategy| (pattern, strategy))
                        .map_err(|e| {
                            SafetensorError::new_err(format!("error parsing shard strategy: {e}"))
                        })
                })
                .collect::<PyResult<HashMap<_, _>>>()?;
            Some(ShardConfig {
                plan: ShardPlan::new(patterns, world_size),
                rank,
            })
        }
        (Some(_), _, _) => {
            return Err(SafetensorError::new_err(
                "rank and world_size must be set for sharding to work",
            ))
        }
        (None, _, _) => {
            return Err(SafetensorError::new_err(
                "cannot set rank or world_size without shard_plan",
            ))
        }
    })
}
