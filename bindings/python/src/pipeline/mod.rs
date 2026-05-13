//! Python bindings for the prefetch pipeline.

use std::collections::VecDeque;

use ionic_rs::numa;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;

use crate::dlpack;
use crate::{get_pydtype, Device, Framework, Open, OpenSources, SafetensorError, TORCH_MODULE};

pub(crate) struct PyIonicError(pub ionic_rs::Error);

impl From<ionic_rs::Error> for PyIonicError {
    fn from(e: ionic_rs::Error) -> Self {
        Self(e)
    }
}

impl From<PyIonicError> for PyErr {
    fn from(e: PyIonicError) -> Self {
        SafetensorError::new_err(e.0.to_string())
    }
}

/// Handle returned by `safe_open.prefetch(...)`. Single-use iterator over
/// `(name, tensor)` pairs; iteration drains and the handle releases each
/// tensor as it's yielded (peak memory ≈ 1× model size). Iteration order
/// is not specified; tensors are yielded as they become ready.
#[pyclass]
pub struct PrefetchHandle {
    pending: VecDeque<(String, Py<PyAny>)>,
}

impl PrefetchHandle {
    /// Dispatch on `(framework, device)`; falls back to eager `get_tensor`.
    pub fn build(open: &Open, names: Vec<String>) -> PyResult<Self> {
        match (&open.framework, &open.device) {
            (Framework::Pytorch, Device::Cuda(ordinal)) => {
                Self::build_cuda(open, names, *ordinal as i32)
            }
            _ => Self::build_trivial(open, names),
        }
    }

    pub fn build_sources(open: &OpenSources, names: Vec<String>) -> PyResult<Self> {
        match (&open.framework, &open.device) {
            (Framework::Pytorch, Device::Cuda(ordinal)) => {
                Self::build_cuda_sources(open, names, *ordinal as i32)
            }
            (_, Device::Cpu) => Err(SafetensorError::new_err(
                "multi-source prefetch on CPU is not implemented yet \
                 (use safe_open + get_tensor for now)",
            )),
            _ => Err(SafetensorError::new_err(format!(
                "multi-source prefetch: unsupported framework/device combo \
                 ({:?} on {:?})",
                open.framework, open.device
            ))),
        }
    }

    fn build_cuda_sources(open: &OpenSources, names: Vec<String>, ordinal: i32) -> PyResult<Self> {
        Python::attach(|py| -> PyResult<Self> {
            // XXX: pinning is best effort, we ignore the result
            let _node = numa::bind_to_gpu_node(ordinal);
            let torch = TORCH_MODULE
                .get()
                .ok_or_else(|| SafetensorError::new_err("torch module not initialized"))?
                .bind(py);

            let source_paths: Vec<std::path::PathBuf> =
                open.sources.iter().map(|s| s.filename.clone()).collect();
            let (mut pipeline, fd_indices) = build_pipeline(py, ordinal, &source_paths)?;

            let mut bufs: Vec<ionic_rs::cuda::DeviceBuf> = Vec::with_capacity(names.len());
            let mut shapes: Vec<Vec<i64>> = Vec::with_capacity(names.len());
            let mut dtypes: Vec<safetensors::Dtype> = Vec::with_capacity(names.len());
            let mut segments: Vec<ionic_rs::iouring::LogicalSegment> =
                Vec::with_capacity(names.len());

            for (i, name) in names.iter().enumerate() {
                let (sidx, info) = open.info(name).ok_or_else(|| {
                    SafetensorError::new_err(format!("tensor '{name}' not in any source"))
                })?;
                let source = &open.sources[sidx];
                let nbytes = info.data_offsets.1 - info.data_offsets.0;
                let buf = pipeline.alloc_device_buf(nbytes).map_err(PyIonicError)?;
                let dst_ptr = buf.as_device_ptr() as usize;
                let from = (info.data_offsets.0 + source.offset) as u64;
                let to = (info.data_offsets.1 + source.offset) as u64;
                segments.push(ionic_rs::iouring::LogicalSegment {
                    fd_idx: fd_indices[sidx],
                    from,
                    to,
                    dst_offset: dst_ptr,
                    user_data: i as u64,
                });
                bufs.push(buf);
                shapes.push(info.shape.iter().map(|&n| n as i64).collect());
                dtypes.push(info.dtype);
            }

            py.detach(|| pipeline.run(&segments))
                .map_err(PyIonicError)?;

            let pending = dlpack_wrap(py, torch, ordinal, names, bufs, shapes, dtypes)?;
            Ok(Self { pending })
        })
    }

    fn build_trivial(open: &Open, names: Vec<String>) -> PyResult<Self> {
        let mut pending = VecDeque::with_capacity(names.len());
        for name in names {
            let t = open.get_tensor(&name)?;
            pending.push_back((name, t));
        }
        Ok(Self { pending })
    }

    fn build_cuda(open: &Open, names: Vec<String>, ordinal: i32) -> PyResult<Self> {
        Python::attach(|py| -> PyResult<Self> {
            // XXX: pinning is best effort, we ignore the result
            let _node = numa::bind_to_gpu_node(ordinal);
            let torch = TORCH_MODULE
                .get()
                .ok_or_else(|| SafetensorError::new_err("torch module not initialized"))?
                .bind(py);

            let paths = [open.filename.clone()];
            let (mut pipeline, _fd_indices) = build_pipeline(py, ordinal, &paths)?;

            let mut bufs: Vec<ionic_rs::cuda::DeviceBuf> = Vec::with_capacity(names.len());
            let mut shapes: Vec<Vec<i64>> = Vec::with_capacity(names.len());
            let mut dtypes: Vec<safetensors::Dtype> = Vec::with_capacity(names.len());
            let mut segments: Vec<ionic_rs::iouring::LogicalSegment> =
                Vec::with_capacity(names.len());

            for (i, name) in names.iter().enumerate() {
                let info = open.metadata.info(name).ok_or_else(|| {
                    SafetensorError::new_err(format!("File does not contain tensor {name}"))
                })?;
                let nbytes = info.data_offsets.1 - info.data_offsets.0;
                let buf = pipeline.alloc_device_buf(nbytes).map_err(PyIonicError)?;
                let dst_ptr = buf.as_device_ptr() as usize;

                let from = (info.data_offsets.0 + open.offset) as u64;
                let to = (info.data_offsets.1 + open.offset) as u64;
                segments.push(ionic_rs::iouring::LogicalSegment {
                    fd_idx: 0,
                    from,
                    to,
                    dst_offset: dst_ptr,
                    user_data: i as u64,
                });
                bufs.push(buf);
                shapes.push(info.shape.iter().map(|&n| n as i64).collect());
                dtypes.push(info.dtype);
            }

            py.detach(|| pipeline.run(&segments))
                .map_err(PyIonicError)?;

            let pending = dlpack_wrap(py, torch, ordinal, names, bufs, shapes, dtypes)?;
            Ok(Self { pending })
        })
    }
}

fn build_pipeline(
    py: Python<'_>,
    ordinal: i32,
    paths: &[std::path::PathBuf],
) -> Result<(ionic_rs::pipeline::CudaPipeline, Vec<u32>), PyIonicError> {
    py.detach(|| {
        let mut p = ionic_rs::pipeline::CudaPipeline::new(
            ordinal,
            ionic_rs::iouring::DEFAULT_QUEUE_DEPTH,
            ionic_rs::iouring::DEFAULT_CHUNK_BYTES,
        )?;
        let mut fds = Vec::with_capacity(paths.len());
        for path in paths {
            fds.push(p.register_file(path)?);
        }
        Ok::<_, ionic_rs::Error>((p, fds))
    })
    .map_err(PyIonicError)
}

fn dlpack_wrap(
    py: Python<'_>,
    torch: &Bound<'_, pyo3::types::PyModule>,
    ordinal: i32,
    names: Vec<String>,
    bufs: Vec<ionic_rs::cuda::DeviceBuf>,
    shapes: Vec<Vec<i64>>,
    dtypes: Vec<safetensors::Dtype>,
) -> PyResult<VecDeque<(String, Py<PyAny>)>> {
    let from_dlpack = torch.getattr(intern!(py, "from_dlpack"))?;
    let device = dlpack::cuda_device(ordinal);
    let mut pending: VecDeque<(String, Py<PyAny>)> = VecDeque::with_capacity(names.len());
    for ((name, buf), (shape, dtype)) in names
        .into_iter()
        .zip(bufs)
        .zip(shapes.into_iter().zip(dtypes))
    {
        let dl_dtype = dlpack::dtype_to_dlpack(dtype);
        let capsule = dlpack::to_capsule(py, buf, shape, dl_dtype, device)?;
        let tensor = from_dlpack.call1((capsule,))?;
        pending.push_back((name, tensor.into_pyobject(py)?.unbind().into_any()));
    }
    Ok(pending)
}

#[pymethods]
impl PrefetchHandle {
    pub fn __len__(&self) -> usize {
        self.pending.len()
    }

    pub fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    pub fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<(String, Py<PyAny>)> {
        slf.pending.pop_front()
    }
}

pub(crate) fn register(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_class::<PrefetchHandle>()?;
    Ok(())
}
