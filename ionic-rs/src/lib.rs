//! ionic-rs: Rust port of Morgan Funtowicz's `ionic` C library.
//!
//! io_uring engine with registered pinned staging buffers, DMA worker on a
//! dedicated CUDA stream, GPU buffer as the sink.
//!
//! Linux-only (`io_uring` is a Linux kernel interface). CUDA is discovered
//! at runtime via `libloading`, so a single build works with or without an
//! Nvidia driver; CUDA primitives return [`Error::CudaUnavailable`] when
//! the driver is absent.

#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

pub mod cuda;
pub mod error;
pub mod iouring;
pub mod numa;
pub mod pipeline;

pub use error::{Error, Result};
