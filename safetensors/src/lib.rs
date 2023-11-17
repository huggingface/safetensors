#![deny(missing_docs)]
#![doc = include_str!("../README.md")]
pub mod slice;
pub mod tensor;
pub use tensor::{serialize, serialize_to_file, Dtype, SafeTensorError, SafeTensors, View};
