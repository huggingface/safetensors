# Safetensors

This repository implements a new simple format for storing tensors
safely (as opposed to pickle) and that is still fast (zero-copy). 


## Yet another format ?

The main rationale for this crate is to remove the need to use
`pickle` on `PyTorch` which is used by default.


Let's take a look at alternatives.

| Format | Safe\* | Zero-copy | Not file size limit | (B)Float-16 support | Flexibility |
| --- | --- | --- | --- | --- | --- |
| pickle (PyTorch) |
| H5 (Tensorflow) |
| SavedModel (Tensorflow) |
| MsgPack (flax) |
| Protobuf (ONNX) |
| Cap'n'Proto |
| SafeTensors | 🗸 | 🗸 | 🗸 | 🗸 | ✗ | 


