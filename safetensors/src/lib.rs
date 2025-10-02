#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]
#![doc = include_str!("../README.md")]

// Always use alloc for consistency between std and no_std

#[macro_use]
extern crate alloc;

pub mod slice;
pub mod tensor;

/// serialize_to_file only valid in std
#[cfg(feature = "std")]
pub use tensor::serialize_to_file;

pub use tensor::{serialize, Dtype, SafeTensorError, SafeTensors, View};
