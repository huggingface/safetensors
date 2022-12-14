#![deny(missing_docs)]
//!# Safetensors
//!
//!This repository implements a new simple format for storing tensors
//!safely (as opposed to pickle) and that is still fast (zero-copy).
//!
//!## Installation
//!### Pip
//!
//! You can install safetensors via the pip manager:
//!
//! ```bash
//! pip install safetensors
//! ```
//!
//!### From source
//!
//!For the sources, you need Rust
//!
//!```bash
//!# Install Rust
//!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
//!# Make sure it's up to date and using stable channel
//!rustup update
//!git clone https://github.com/huggingface/safetensors
//!cd safetensors/bindings/python
//!pip install setuptools_rust
//!pip install -e .
//!```
//!
//!## Getting started
//!
//!```python
//!from safetensors import safe_open
//!from safetensors.torch import save_file
//!
//!tensors = {
//!    "weight1": torch.zeros((1024, 1024)),
//!    "weight2": torch.zeros((1024, 1024))
//!}
//!save_file(tensors, "model.safetensors")
//!
//!tensors = {}
//!with safe_open("model.safetensors", framework="pt", device="cpu") as f:
//!    for key in f.keys():
//!        tensors[key] = f.get_tensor(key)
//!```
//!
//![Python documentation](https://github.com/huggingface/safetensors/tree/main/bindings/python)
//!
//!
//!## Format
//!
//! - 8 bytes: `N`, a u64 int, containing the size of the header
//! - N bytes: a JSON utf-8 string representing the header.
//!   - The header is a dict like {"TENSOR_NAME": {"dtype": "float16", "shape": [1, 16, 256], "offsets": (X, Y)}}, where X and Y are the offsets in the byte buffer of the tensor data
//!   - A special key `__metadata__` is allowed to contain free form text map.
//! - Rest of the file: byte-buffer.
//!
//!
//!## Yet another format ?
//!
//!The main rationale for this crate is to remove the need to use
//!`pickle` on `PyTorch` which is used by default.
//!There are other formats out there used by machine learning and more general
//!formats.
//!
//!
//!Let's take a look at alternatives and why this format is deemed interesting.
//!This is my very personal and probably biased view:
//!
//!| Format                  | Safe | Zero-copy | Lazy loading | No file size limit | Layout control | Flexibility | Bfloat16
//!| ----------------------- | --- | --- | --- | --- | --- | --- | --- |
//!| pickle (PyTorch)        | ✗ | ✗ | ✗ | 🗸 | ✗ | 🗸 | 🗸 |
//!| H5 (Tensorflow)         | 🗸 | ✗ | 🗸 | 🗸 | ~ | ~ | ✗ |
//!| SavedModel (Tensorflow) | 🗸 | ✗ | ✗ | 🗸 | 🗸 | ✗ | 🗸 |
//!| MsgPack (flax)          | 🗸 | 🗸 | ✗ | 🗸 | ✗ | ✗ | 🗸 |
//!| Protobuf (ONNX)         | 🗸 | ✗ | ✗ | ✗ | ✗ | ✗ | 🗸 |
//!| Cap'n'Proto             | 🗸 | 🗸 | ~ | 🗸 | 🗸 | ~ | ✗ |
//!| Arrow                   | ? | ? | ? | ? | ? | ? | ✗ |
//!| Numpy (npy,npz)         | 🗸 | ? | ? | ✗ | 🗸 | ✗ | ✗ |
//!| SafeTensors             | 🗸 | 🗸 | 🗸 | 🗸 | 🗸 | ✗ | 🗸 |
//!
//!- Safe: Can I use a file randomly downloaded and expect not to run arbitrary code ?
//!- Zero-copy: Does reading the file require more memory than the original file ?
//!- Lazy loading: Can I inspect the file without loading everything ? And loading only
//!some tensors in it without scanning the whole file (distributed setting) ?
//!- Layout control: Lazy loading, is not necessarily enough since if the information about tensors is spread out in your file, then even if the information is lazily accessible you might have to access most of your file to read the available tensors (incurring many DISK -> RAM copies). Controlling the layout to keep fast access to single tensors is important.
//!- No file size limit: Is there a limit to the file size ?
//!- Flexibility: Can I save custom code in the format and be able to use it later with zero extra code ? (~ means we can store more than pure tensors, but no custom code)
//!- Bfloat16: Does the format support native bfloat16 (meaning no weird workarounds are
//!necessary)? This is becoming increasingly important in the ML world.
//!
//!
//!## Main oppositions
//!
//!- Pickle: Unsafe, runs arbitrary code
//!- H5: Apparently now discouraged for TF/Keras. Seems like a great fit otherwise actually. Some classic use after free issues: <https://www.cvedetails.com/vulnerability-list/vendor_id-15991/product_id-35054/Hdfgroup-Hdf5.html>. On a very different level than pickle security-wise. Also 210k lines of code vs ~400 lines for this lib currently.
//!- SavedModel: Tensorflow specific (it contains TF graph information).
//!- MsgPack: No layout control to enable lazy loading (important for loading specific parts in distributed setting)
//!- Protobuf: Hard 2Go max file size limit
//!- Cap'n'proto: Float16 support is not present [link](https://capnproto.org/language.html#built-in-types) so using a manual wrapper over a byte-buffer would be necessary. Layout control seems possible but not trivial as buffers have limitations [link](https://stackoverflow.com/questions/48458839/capnproto-maximum-filesize).
//!- Numpy (npz): No `bfloat16` support. Vulnerable to zip bombs (DOS).
//!- Arrow: No `bfloat16` support. Seem to require decoding [link](https://arrow.apache.org/docs/python/parquet.html#reading-parquet-and-memory-mapping)
//!
//!## Notes
//!
//!- Zero-copy: No format is really zero-copy in ML, it needs to go from disk to RAM/GPU RAM (that takes time). Also
//!    In PyTorch/numpy, you need a mutable buffer, and we don't really want to mutate a mmaped file, so 1 copy is really necessary to use the thing freely in user code. That being said, zero-copy is achievable in Rust if it's wanted and safety can be guaranteed by some other means.
//!    SafeTensors is not zero-copy for the header. The choice of JSON is pretty arbitrary, but since deserialization is <<< of the time required to load the actual tensor data and is readable I went that way, (also space is <<< to the tensor data).
//!
//!- Endianness: Little-endian. This can be modified later, but it feels really unecessary at the
//!moment.
//!- Order: 'C' or row-major. This seems to have won. We can add that information later if needed.
//!- Stride: No striding, all tensors need to be packed before being serialized. I have yet to see a case where it seems useful to have a strided tensor stored in serialized format.
//!
//!## Benefits
//!
//!Since we can invent a new format we can propose additional benefits:
//!
//!- Prevent DOS attacks: We can craft the format in such a way that it's almost
//!impossible to use malicious files to DOS attack a user. Currently, there's a limit
//!on the size of the header of 100MB to prevent parsing extremely large JSON.
//!  Also when reading the file, there's a guarantee that addresses in the file
//!  do not overlap in any way, meaning when you're loading a file you should never
//!  exceed the size of the file in memory
//!
//!- Faster load: PyTorch seems to be the fastest file to load out in the major
//! ML formats. However, it does seem to have an extra copy on CPU, which we
//! can bypass in this lib [link](https://github.com/huggingface/safetensors/pull/33).
//! Currently, CPU loading times are extremely fast with this lib compared to pickle.
//! GPU loading times can be sped up but are still hidden behind an environment variable
//! (`SAFETENSORS_FAST_GPU=1`) because it hasn't received enough external scrutiny to be safe.
//! But it does load roughly 2X faster than PyTorch on regular Linux hardware because of this extra CPU copy skip.
//!
//!- Lazy loading: in distributed (multi-node or multi-gpu) settings, it's nice to be able to
//!load only part of the tensors on the various models. For
//![BLOOM](https://huggingface.co/bigscience/bloom) using this format enabled
//!to load the model on 8 GPUs from 10mn with regular PyTorch weights down to 45s.
//!This really speeds up feedbacks loops when developing on the model. For instance
//!you don't have to have separate copies of the weights when changing the distribution
//!strategy (for instance Pipeline Parallelism vs Tensor Parallelism).

pub mod slice;
pub mod tensor;
