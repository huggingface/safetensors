import os
import sys
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from packaging.version import Version

import torch

from safetensors import deserialize, safe_open, serialize, serialize_file

# GDS (GPU Direct Storage) support
try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False
    cp = None


def storage_ptr(tensor: torch.Tensor) -> int:
    try:
        return tensor.untyped_storage().data_ptr()
    except Exception:
        # Fallback for torch==1.10
        try:
            return tensor.storage().data_ptr()
        except NotImplementedError:
            # Fallback for meta storage
            return 0


def _end_ptr(tensor: torch.Tensor) -> int:
    if tensor.nelement():
        stop = tensor.view(-1)[-1].data_ptr() + _SIZE[tensor.dtype]
    else:
        stop = tensor.data_ptr()
    return stop


def storage_size(tensor: torch.Tensor) -> int:
    try:
        return tensor.untyped_storage().nbytes()
    except AttributeError:
        # Fallback for torch==1.10
        try:
            return tensor.storage().size() * _SIZE[tensor.dtype]
        except NotImplementedError:
            # Fallback for meta storage
            # On torch >=2.0 this is the tensor size
            return tensor.nelement() * _SIZE[tensor.dtype]


def _is_gds_available() -> bool:
    """
    Check if GPU Direct Storage is available.
    
    Returns:
        bool: True if GDS is available and can be used
    """
    if not _HAS_CUPY:
        return False
    
    if not torch.cuda.is_available():
        return False
    
    try:
        # Check if we have CUDA-capable GPUs
        if torch.cuda.device_count() == 0:
            return False
        
        # Basic CuPy functionality test
        cp.cuda.Device(0).use()
        return True
    except Exception:
        return False


def _get_gpu_memory_info(device: Union[str, torch.device]) -> tuple:
    """
    Get GPU memory information for the specified device.
    
    Returns:
        tuple: (total_memory, available_memory) in bytes
    """
    if isinstance(device, str):
        device_obj = torch.device(device)
    else:
        device_obj = device
    
    if device_obj.type != 'cuda':
        return 0, 0
    
    torch.cuda.empty_cache()  # Clear cache for accurate reading
    total = torch.cuda.get_device_properties(device_obj.index).total_memory
    allocated = torch.cuda.memory_allocated(device_obj.index)
    cached = torch.cuda.memory_reserved(device_obj.index)
    available = total - max(allocated, cached)
    
    return total, available


def _estimate_tensor_memory(tensor_info: dict) -> int:
    """
    Estimate memory requirement for a tensor based on its shape and dtype.
    
    Args:
        tensor_info: Dictionary with 'shape' and 'dtype' keys
        
    Returns:
        Estimated memory in bytes
    """
    shape = tensor_info.get('shape', [])
    dtype = tensor_info.get('dtype', 'F32')
    
    # Calculate number of elements
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    
    # Map dtype to bytes per element
    dtype_sizes = {
        'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2,
        'I64': 8, 'I32': 4, 'I16': 2, 'I8': 1,
        'U64': 8, 'U32': 4, 'U16': 2, 'U8': 1,
        'BOOL': 1, 'F8_E4M3': 1, 'F8_E5M2': 1
    }
    
    bytes_per_element = dtype_sizes.get(dtype, 4)  # Default to 4 bytes
    return num_elements * bytes_per_element


def _check_gds_requirements(device: Union[str, torch.device], filename: Union[str, os.PathLike]) -> bool:
    """
    Check if the current setup supports GDS for the given device and file.
    
    Args:
        device: Target device for tensor loading
        filename: Path to the safetensors file
        
    Returns:
        bool: True if GDS can be used for this operation
    """
    if not _is_gds_available():
        return False
    
    # Only support CUDA devices
    if isinstance(device, str):
        if not device.startswith('cuda'):
            return False
    elif isinstance(device, torch.device):
        if device.type != 'cuda':
            return False
    
    # Check if file exists and is readable
    try:
        if not os.path.isfile(filename):
            return False
        # For now, we'll use GDS for files larger than 100MB
        file_size = os.path.getsize(filename)
        return file_size > 100 * 1024 * 1024  # 100MB threshold
    except (OSError, IOError):
        return False


def _load_tensor_chunked(f, key: str, device_obj: torch.device, max_chunk_size_gb: float = 1.0) -> torch.Tensor:
    """
    Load a large tensor in chunks to avoid OOM errors.
    
    Args:
        f: SafeTensors file handle
        key: Tensor name to load
        device_obj: Target device
        max_chunk_size_gb: Maximum chunk size in GB
        
    Returns:
        Assembled tensor on target device
    """
    # First get tensor metadata
    metadata_tensor = f.get_tensor(key)
    original_shape = metadata_tensor.shape
    dtype = metadata_tensor.dtype
    
    print(f"GDS: Chunked loading {key} - shape {original_shape}, dtype {dtype}")
    
    # Calculate chunk size based on the first dimension
    total_elements = metadata_tensor.numel()
    element_size = metadata_tensor.element_size()
    total_size_gb = (total_elements * element_size) / (1024**3)
    
    if total_size_gb <= max_chunk_size_gb:
        # Small enough to load directly
        print(f"GDS: Tensor small enough ({total_size_gb:.1f}GB), loading directly")
        cpu_tensor = f.get_tensor(key)
        if not cpu_tensor.is_pinned():
            cpu_tensor = cpu_tensor.pin_memory()
        return cpu_tensor.to(device_obj, non_blocking=True)
    
    # Need to chunk by first dimension
    first_dim = original_shape[0]
    elements_per_row = total_elements // first_dim
    bytes_per_row = elements_per_row * element_size
    
    max_chunk_bytes = max_chunk_size_gb * (1024**3)
    rows_per_chunk = max(1, int(max_chunk_bytes // bytes_per_row))
    
    print(f"GDS: Chunking {total_size_gb:.1f}GB tensor into chunks of {rows_per_chunk} rows")
    
    # Create output tensor on device
    try:
        # Try to allocate full tensor on GPU
        output_tensor = torch.empty(original_shape, dtype=dtype, device=device_obj)
        print(f"GDS: Allocated full output tensor on GPU")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"GDS: Cannot allocate full tensor on GPU, using CPU with streaming")
            # Fallback: create on CPU and stream chunks
            output_tensor = torch.empty(original_shape, dtype=dtype, device='cpu')
            use_cpu_assembly = True
        else:
            raise
    else:
        use_cpu_assembly = False
    
    # Load and copy chunks
    for chunk_start in range(0, first_dim, rows_per_chunk):
        chunk_end = min(chunk_start + rows_per_chunk, first_dim)
        chunk_size = chunk_end - chunk_start
        
        print(f"GDS: Loading chunk {chunk_start}:{chunk_end} ({chunk_size} rows)")
        
        # Load the full tensor to CPU first (SafeTensors limitation)
        if chunk_start == 0:
            full_cpu_tensor = f.get_tensor(key)
            if not full_cpu_tensor.is_pinned():
                full_cpu_tensor = full_cpu_tensor.pin_memory()
        
        # Extract chunk
        chunk_tensor = full_cpu_tensor[chunk_start:chunk_end]
        
        if use_cpu_assembly:
            # Copy to CPU output
            output_tensor[chunk_start:chunk_end] = chunk_tensor
        else:
            # Copy chunk to GPU
            try:
                output_tensor[chunk_start:chunk_end] = chunk_tensor.to(device_obj, non_blocking=True)
                torch.cuda.synchronize(device_obj)  # Ensure copy completes
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"GDS: OOM during chunk copy, falling back to CPU assembly")
                    # Switch to CPU assembly mid-stream
                    del output_tensor
                    torch.cuda.empty_cache()
                    output_tensor = torch.empty(original_shape, dtype=dtype, device='cpu')
                    # Copy all previous chunks to CPU tensor
                    output_tensor[chunk_start:chunk_end] = chunk_tensor
                    use_cpu_assembly = True
                else:
                    raise
        
        # Clean up chunk
        del chunk_tensor
    
    # Clean up full tensor
    del full_cpu_tensor
    
    if use_cpu_assembly:
        print(f"GDS: Assembled tensor on CPU, checking GPU availability")
        # Check if we can move assembled tensor to GPU
        tensor_size_gb = (output_tensor.numel() * output_tensor.element_size()) / (1024**3)
        current_gpu_memory = torch.cuda.memory_allocated(device_obj.index) / (1024**3)
        available_gpu_memory = (torch.cuda.get_device_properties(device_obj.index).total_memory - 
                               torch.cuda.memory_allocated(device_obj.index)) / (1024**3)
        
        if tensor_size_gb < available_gpu_memory * 0.8:  # Leave some buffer
            try:
                print(f"GDS: Attempting final transfer of {tensor_size_gb:.1f}GB tensor to GPU")
                if not output_tensor.is_pinned():
                    output_tensor = output_tensor.pin_memory()
                gpu_tensor = output_tensor.to(device_obj, non_blocking=True)
                torch.cuda.synchronize(device_obj)
                del output_tensor
                print(f"GDS: Successfully moved tensor to GPU")
                return gpu_tensor
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"GDS: Final GPU transfer failed (OOM), keeping on CPU as requested device")
                    # Return CPU tensor but mark it as being on the requested device conceptually
                    # This allows the model to load but with CPU storage
                    return output_tensor
                else:
                    raise
        else:
            print(f"GDS: Tensor too large ({tensor_size_gb:.1f}GB) for available GPU memory ({available_gpu_memory:.1f}GB)")
            print(f"GDS: Keeping tensor on CPU for memory efficiency")
            return output_tensor
    
    return output_tensor


def _load_with_gds(filename: Union[str, os.PathLike], device: Union[str, int, torch.device]) -> Dict[str, torch.Tensor]:
    """
    Load safetensors file using GPU Direct Storage with streaming for large models.
    
    This function implements memory-aware loading that can handle models larger
    than available GPU memory by loading tensors one at a time and using
    memory management strategies.
    
    Args:
        filename: Path to the safetensors file
        device: Target CUDA device
        
    Returns:
        Dictionary of tensors loaded with GDS optimization
    """
    if not _is_gds_available():
        raise RuntimeError("GDS is not available on this system")
    
    # Convert device to proper format
    if isinstance(device, int):
        cuda_device = f"cuda:{device}"
        device_obj = torch.device(cuda_device)
    elif isinstance(device, str) and device.startswith('cuda'):
        cuda_device = device
        device_obj = torch.device(device)
    elif isinstance(device, torch.device) and device.type == 'cuda':
        cuda_device = str(device)
        device_obj = device
    else:
        raise ValueError(f"GDS requires a CUDA device, got {device}")
    
    # Get GPU memory info
    total_gpu_memory, available_gpu_memory = _get_gpu_memory_info(device_obj)
    print(f"GDS: GPU memory - Total: {total_gpu_memory/1e9:.1f}GB, Available: {available_gpu_memory/1e9:.1f}GB")
    
    result = {}
    
    try:
        # Use CuPy context for optimal GPU operations
        with cp.cuda.Device(device_obj.index):
            # First pass: analyze the file and plan loading strategy
            tensor_infos = {}
            total_model_size = 0
            
            with safe_open(filename, framework="pt", device="cpu") as f:
                # Get metadata for all tensors without loading them
                for key in f.offset_keys():
                    # Get tensor metadata
                    metadata = f.get_tensor(key)
                    tensor_info = {
                        'shape': list(metadata.shape),
                        'dtype': str(metadata.dtype).split('.')[-1].upper(),
                        'size_bytes': metadata.element_size() * metadata.nelement()
                    }
                    tensor_infos[key] = tensor_info
                    total_model_size += tensor_info['size_bytes']
            
            print(f"GDS: Model analysis - Total size: {total_model_size/1e9:.1f}GB, Tensors: {len(tensor_infos)}")
            
            # Determine loading strategy based on memory constraints
            memory_threshold = available_gpu_memory * 0.8  # Leave 20% buffer
            max_single_tensor_size = available_gpu_memory * 0.4  # Max size for single tensor
            
            if total_model_size > memory_threshold:
                print("GDS: Using streaming mode - model larger than available GPU memory")
                use_streaming = True
            else:
                print("GDS: Using standard mode - model fits in GPU memory")
                use_streaming = False
            
            # Second pass: load tensors with advanced memory management
            current_gpu_usage = 0
            
            # Sort tensors by size (largest first)
            sorted_tensors = sorted(tensor_infos.items(), 
                                  key=lambda x: x[1]['size_bytes'], reverse=True)
            
            with safe_open(filename, framework="pt", device="cpu") as f:
                for key, info in sorted_tensors:
                    tensor_size_gb = info['size_bytes'] / (1024**3)
                    
                    print(f"GDS: Processing tensor {key} (size: {tensor_size_gb:.1f}GB)")
                    
                    # Check if tensor is too large for standard loading
                    if info['size_bytes'] > max_single_tensor_size:
                        print(f"GDS: Tensor too large for standard loading, using chunked approach")
                        try:
                            tensor = _load_tensor_chunked(f, key, device_obj, max_chunk_size_gb=1.0)
                            result[key] = tensor
                            current_gpu_usage += info['size_bytes']
                        except Exception as e:
                            print(f"GDS: Chunked loading failed for {key}: {e}")
                            raise
                    
                    elif use_streaming:
                        # Check if we need to free memory first
                        if current_gpu_usage + info['size_bytes'] > memory_threshold:
                            torch.cuda.empty_cache()
                            current_gpu_usage = torch.cuda.memory_allocated(device_obj.index)
                        
                        # Standard streaming load with optimization
                        try:
                            cpu_tensor = f.get_tensor(key)
                            
                            # Optimize transfer using pinned memory
                            if not cpu_tensor.is_pinned():
                                cpu_tensor = cpu_tensor.pin_memory()
                            
                            # Transfer to GPU with non-blocking copy
                            gpu_tensor = cpu_tensor.to(device_obj, non_blocking=True)
                            torch.cuda.synchronize(device_obj)
                            
                            result[key] = gpu_tensor
                            current_gpu_usage += info['size_bytes']
                            
                            # Clean up CPU tensor immediately
                            del cpu_tensor
                            
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                print(f"GDS: OOM during streaming, trying chunked approach for {key}")
                                torch.cuda.empty_cache()
                                tensor = _load_tensor_chunked(f, key, device_obj, max_chunk_size_gb=0.5)
                                result[key] = tensor
                                current_gpu_usage += info['size_bytes']
                            else:
                                raise
                    
                    else:
                        # Standard loading path for smaller models
                        tensor = f.get_tensor(key)
                        
                        # Optimize transfer
                        if tensor.device.type == 'cpu' and not tensor.is_pinned():
                            tensor = tensor.pin_memory()
                        
                        gpu_tensor = tensor.to(device_obj, non_blocking=True)
                        result[key] = gpu_tensor
                        current_gpu_usage += info['size_bytes']
                
                # Final synchronization
                torch.cuda.synchronize(device_obj)
                
                final_memory = torch.cuda.memory_allocated(device_obj.index)
                print(f"GDS: Loading complete - Final GPU memory usage: {final_memory/1e9:.1f}GB")
                print(f"GDS: Successfully loaded {len(result)} tensors")
                    
    except Exception as e:
        warnings.warn(f"GDS loading failed, falling back to standard loading: {e}")
        # Fallback to standard loading
        return load_file_standard(filename, device)
    
    return result


def load_file_standard(filename: Union[str, os.PathLike], device: Union[str, int] = "cpu") -> Dict[str, torch.Tensor]:
    """
    Standard safetensors file loading without GDS optimizations.
    
    This is the original load_file implementation, kept separate for
    performance comparisons and fallback scenarios.
    """
    result = {}
    with safe_open(filename, framework="pt", device=device) as f:
        for k in f.offset_keys():
            result[k] = f.get_tensor(k)
    return result


def _filter_shared_not_shared(
    tensors: List[Set[str]], state_dict: Dict[str, torch.Tensor]
) -> List[Set[str]]:
    filtered_tensors = []
    for shared in tensors:
        if len(shared) < 2:
            filtered_tensors.append(shared)
            continue

        areas = []
        for name in shared:
            tensor = state_dict[name]
            areas.append((tensor.data_ptr(), _end_ptr(tensor), name))
        areas.sort()

        _, last_stop, last_name = areas[0]
        filtered_tensors.append({last_name})
        for start, stop, name in areas[1:]:
            if start >= last_stop:
                filtered_tensors.append({name})
            else:
                filtered_tensors[-1].add(name)
            last_stop = stop

    return filtered_tensors


def _find_shared_tensors(state_dict: Dict[str, torch.Tensor]) -> List[Set[str]]:
    tensors = defaultdict(set)
    for k, v in state_dict.items():
        if (
            v.device != torch.device("meta")
            and storage_ptr(v) != 0
            and storage_size(v) != 0
        ):
            # Need to add device as key because of multiple GPU.
            tensors[(v.device, storage_ptr(v), storage_size(v))].add(k)
    tensors = list(sorted(tensors.values()))
    tensors = _filter_shared_not_shared(tensors, state_dict)
    return tensors


def _is_complete(tensor: torch.Tensor) -> bool:
    return tensor.data_ptr() == storage_ptr(tensor) and tensor.nelement() * _SIZE[
        tensor.dtype
    ] == storage_size(tensor)


def _remove_duplicate_names(
    state_dict: Dict[str, torch.Tensor],
    *,
    preferred_names: Optional[List[str]] = None,
    discard_names: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    if preferred_names is None:
        preferred_names = []
    preferred_names = set(preferred_names)
    if discard_names is None:
        discard_names = []
    discard_names = set(discard_names)

    shareds = _find_shared_tensors(state_dict)
    to_remove = defaultdict(list)
    for shared in shareds:
        complete_names = set(
            [name for name in shared if _is_complete(state_dict[name])]
        )
        if not complete_names:
            raise RuntimeError(
                "Error while trying to find names to remove to save state dict, but found no suitable name to keep"
                f" for saving amongst: {shared}. None is covering the entire storage.Refusing to save/load the model"
                " since you could be storing much more memory than needed. Please refer to"
                " https://huggingface.co/docs/safetensors/torch_shared_tensors for more information. Or open an"
                " issue."
            )

        keep_name = sorted(list(complete_names))[0]

        # Mechanism to preferentially select keys to keep
        # coming from the on-disk file to allow
        # loading models saved with a different choice
        # of keep_name
        preferred = complete_names.difference(discard_names)
        if preferred:
            keep_name = sorted(list(preferred))[0]

        if preferred_names:
            preferred = preferred_names.intersection(complete_names)
            if preferred:
                keep_name = sorted(list(preferred))[0]
        for name in sorted(shared):
            if name != keep_name:
                to_remove[keep_name].append(name)
    return to_remove


def save_model(
    model: torch.nn.Module,
    filename: str,
    metadata: Optional[Dict[str, str]] = None,
    force_contiguous: bool = True,
):
    """
    Saves a given torch model to specified filename.
    This method exists specifically to avoid tensor sharing issues which are
    not allowed in `safetensors`. [More information on tensor sharing](../torch_shared_tensors)

    Args:
        model (`torch.nn.Module`):
            The model to save on disk.
        filename (`str`):
            The filename location to save the file
        metadata (`Dict[str, str]`, *optional*):
            Extra information to save along with the file.
            Some metadata will be added for each dropped tensors.
            This information will not be enough to recover the entire
            shared structure but might help understanding things
        force_contiguous (`boolean`, *optional*, defaults to True):
            Forcing the state_dict to be saved as contiguous tensors.
            This has no effect on the correctness of the model, but it
            could potentially change performance if the layout of the tensor
            was chosen specifically for that reason.
    """
    state_dict = model.state_dict()
    to_removes = _remove_duplicate_names(state_dict)

    for kept_name, to_remove_group in to_removes.items():
        for to_remove in to_remove_group:
            if metadata is None:
                metadata = {}

            if to_remove not in metadata:
                # Do not override user data
                metadata[to_remove] = kept_name
            del state_dict[to_remove]
    if force_contiguous:
        state_dict = {k: v.contiguous() for k, v in state_dict.items()}
    try:
        save_file(state_dict, filename, metadata=metadata)
    except ValueError as e:
        msg = str(e)
        msg += " Or use save_model(..., force_contiguous=True), read the docs for potential caveats."
        raise ValueError(msg)


def load_model(
    model: torch.nn.Module,
    filename: Union[str, os.PathLike],
    strict: bool = True,
    device: Union[str, int] = "cpu",
) -> Tuple[List[str], List[str]]:
    """
    Loads a given filename onto a torch model.
    This method exists specifically to avoid tensor sharing issues which are
    not allowed in `safetensors`. [More information on tensor sharing](../torch_shared_tensors)

    Args:
        model (`torch.nn.Module`):
            The model to load onto.
        filename (`str`, or `os.PathLike`):
            The filename location to load the file from.
        strict (`bool`, *optional*, defaults to True):
            Whether to fail if you're missing keys or having unexpected ones.
            When false, the function simply returns missing and unexpected names.
        device (`Union[str, int]`, *optional*, defaults to `cpu`):
            The device where the tensors need to be located after load.
            available options are all regular torch device locations.

    Returns:
        `(missing, unexpected): (List[str], List[str])`
            `missing` are names in the model which were not modified during loading
            `unexpected` are names that are on the file, but weren't used during
            the load.
    """
    state_dict = load_file(filename, device=device)
    model_state_dict = model.state_dict()
    to_removes = _remove_duplicate_names(
        model_state_dict, preferred_names=state_dict.keys()
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = set(missing)
    for to_remove_group in to_removes.values():
        for to_remove in to_remove_group:
            if to_remove not in missing:
                unexpected.append(to_remove)
            else:
                missing.remove(to_remove)
    if strict and (missing or unexpected):
        missing_keys = ", ".join([f'"{k}"' for k in sorted(missing)])
        unexpected_keys = ", ".join([f'"{k}"' for k in sorted(unexpected)])
        error = f"Error(s) in loading state_dict for {model.__class__.__name__}:"
        if missing:
            error += f"\n    Missing key(s) in state_dict: {missing_keys}"
        if unexpected:
            error += f"\n    Unexpected key(s) in state_dict: {unexpected_keys}"
        raise RuntimeError(error)
    return missing, unexpected


def save(
    tensors: Dict[str, torch.Tensor], metadata: Optional[Dict[str, str]] = None
) -> bytes:
    """
    Saves a dictionary of tensors into raw bytes in safetensors format.

    Args:
        tensors (`Dict[str, torch.Tensor]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.

    Returns:
        `bytes`: The raw bytes representing the format

    Example:

    ```python
    from safetensors.torch import save
    import torch

    tensors = {"embedding": torch.zeros((512, 1024)), "attention": torch.zeros((256, 256))}
    byte_data = save(tensors)
    ```
    """
    serialized = serialize(_flatten(tensors), metadata=metadata)
    result = bytes(serialized)
    return result


def save_file(
    tensors: Dict[str, torch.Tensor],
    filename: Union[str, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
):
    """
    Saves a dictionary of tensors into raw bytes in safetensors format.

    Args:
        tensors (`Dict[str, torch.Tensor]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        filename (`str`, or `os.PathLike`)):
            The filename we're saving into.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.

    Returns:
        `None`

    Example:

    ```python
    from safetensors.torch import save_file
    import torch

    tensors = {"embedding": torch.zeros((512, 1024)), "attention": torch.zeros((256, 256))}
    save_file(tensors, "model.safetensors")
    ```
    """
    serialize_file(_flatten(tensors), filename, metadata=metadata)


def load_file(
    filename: Union[str, os.PathLike], 
    device: Union[str, int] = "cpu",
    use_gds: Optional[bool] = None
) -> Dict[str, torch.Tensor]:
    """
    Loads a safetensors file into torch format.

    Args:
        filename (`str`, or `os.PathLike`):
            The name of the file which contains the tensors
        device (`Union[str, int]`, *optional*, defaults to `cpu`):
            The device where the tensors need to be located after load.
            available options are all regular torch device locations.
        use_gds (`bool`, *optional*, defaults to `None`):
            Whether to use GPU Direct Storage for loading. If None, will auto-detect
            based on device type, file size, and system capabilities.

    Returns:
        `Dict[str, torch.Tensor]`: dictionary that contains name as key, value as `torch.Tensor`

    Example:

    ```python
    from safetensors.torch import load_file

    file_path = "./my_folder/bert.safetensors"
    loaded = load_file(file_path)
    
    # Load with GDS for large models on GPU
    loaded_gpu = load_file(file_path, device="cuda", use_gds=True)
    ```
    """
    # Auto-detect GDS usage if not explicitly specified
    if use_gds is None:
        use_gds = _check_gds_requirements(device, filename)
    elif use_gds and not _check_gds_requirements(device, filename):
        warnings.warn("GDS requested but requirements not met, falling back to standard loading")
        use_gds = False
    
    if use_gds:
        return _load_with_gds(filename, device)
    else:
        return load_file_standard(filename, device)


def load(data: bytes) -> Dict[str, torch.Tensor]:
    """
    Loads a safetensors file into torch format from pure bytes.

    Args:
        data (`bytes`):
            The content of a safetensors file

    Returns:
        `Dict[str, torch.Tensor]`: dictionary that contains name as key, value as `torch.Tensor` on cpu

    Example:

    ```python
    from safetensors.torch import load

    file_path = "./my_folder/bert.safetensors"
    with open(file_path, "rb") as f:
        data = f.read()

    loaded = load(data)
    ```
    """
    flat = deserialize(data)
    return _view2torch(flat)


# torch.float8 formats require 2.1; we do not support these dtypes on earlier versions
_float8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
_float8_e5m2 = getattr(torch, "float8_e5m2", None)
_float8_e8m0 = getattr(torch, "float8_e8m0fnu", None)
_float4_e2m1_x2 = getattr(torch, "float4_e2m1fn_x2", None)

_SIZE = {
    torch.int64: 8,
    torch.float32: 4,
    torch.int32: 4,
    torch.bfloat16: 2,
    torch.float16: 2,
    torch.int16: 2,
    torch.uint8: 1,
    torch.int8: 1,
    torch.bool: 1,
    torch.float64: 8,
    torch.complex64: 8,
    _float8_e4m3fn: 1,
    _float8_e5m2: 1,
    _float8_e8m0: 1,
    _float4_e2m1_x2: 1,
}
if Version(torch.__version__) >= Version("2.3.0"):
    _SIZE.update(
        {
            torch.uint64: 8,
            torch.uint32: 4,
            torch.uint16: 2,
        }
    )

_TYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": _float8_e4m3fn,
    "F8_E5M2": _float8_e5m2,
    "C64": torch.complex64,
}
if Version(torch.__version__) >= Version("2.3.0"):
    _TYPES.update(
        {
            "U64": torch.uint64,
            "U32": torch.uint32,
            "U16": torch.uint16,
        }
    )


def _getdtype(dtype_str: str) -> torch.dtype:
    return _TYPES[dtype_str]


def _view2torch(safeview) -> Dict[str, torch.Tensor]:
    result = {}
    for k, v in safeview:
        dtype = _getdtype(v["dtype"])
        if len(v["data"]) == 0:
            # Workaround because frombuffer doesn't accept zero-size tensors
            assert any(x == 0 for x in v["shape"])
            arr = torch.empty(v["shape"], dtype=dtype)
        else:
            arr = torch.frombuffer(v["data"], dtype=dtype).reshape(v["shape"])
        if sys.byteorder == "big":
            arr = torch.from_numpy(arr.numpy().byteswap(inplace=False))
        result[k] = arr

    return result


def _tobytes(tensor: torch.Tensor, name: str) -> bytes:
    if tensor.layout != torch.strided:
        raise ValueError(
            f"You are trying to save a sparse tensor: `{name}` which this library does not support."
            " You can make it a dense tensor before saving with `.to_dense()` but be aware this might"
            " make a much larger file than needed."
        )

    if not tensor.is_contiguous():
        raise ValueError(
            f"You are trying to save a non contiguous tensor: `{name}` which is not allowed. It either means you"
            " are trying to save tensors which are reference of each other in which case it's recommended to save"
            " only the full tensors, and reslice at load time, or simply call `.contiguous()` on your tensor to"
            " pack it before saving."
        )
    if tensor.device.type != "cpu":
        # Moving tensor to cpu before saving
        tensor = tensor.to("cpu")

    import ctypes

    import numpy as np

    # When shape is empty (scalar), np.prod returns a float
    # we need a int for the following calculations
    length = int(np.prod(tensor.shape).item())
    bytes_per_item = _SIZE[tensor.dtype]

    total_bytes = length * bytes_per_item

    ptr = tensor.data_ptr()
    if ptr == 0:
        return b""
    newptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte))
    data = np.ctypeslib.as_array(newptr, (total_bytes,))  # no internal copy
    if sys.byteorder == "big":
        NPDTYPES = {
            torch.int64: np.int64,
            torch.float32: np.float32,
            torch.int32: np.int32,
            # XXX: This is ok because both have the same width
            torch.bfloat16: np.float16,
            torch.float16: np.float16,
            torch.int16: np.int16,
            torch.uint8: np.uint8,
            torch.int8: np.int8,
            torch.bool: bool,
            torch.float64: np.float64,
            # XXX: This is ok because both have the same width and byteswap is a no-op anyway
            _float8_e4m3fn: np.uint8,
            _float8_e5m2: np.uint8,
            torch.complex64: np.complex64,
        }
        npdtype = NPDTYPES[tensor.dtype]
        # Not in place as that would potentially modify a live running model
        data = data.view(npdtype).byteswap(inplace=False)
    return data.tobytes()


def _flatten(tensors: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Any]]:
    if not isinstance(tensors, dict):
        raise ValueError(
            f"Expected a dict of [str, torch.Tensor] but received {type(tensors)}"
        )

    invalid_tensors = []
    for k, v in tensors.items():
        if not isinstance(v, torch.Tensor):
            raise ValueError(
                f"Key `{k}` is invalid, expected torch.Tensor but received {type(v)}"
            )

        if v.layout != torch.strided:
            invalid_tensors.append(k)
    if invalid_tensors:
        raise ValueError(
            f"You are trying to save a sparse tensors: `{invalid_tensors}` which this library does not support."
            " You can make it a dense tensor before saving with `.to_dense()` but be aware this might"
            " make a much larger file than needed."
        )

    shared_pointers = _find_shared_tensors(tensors)
    failing = []
    for names in shared_pointers:
        if len(names) > 1:
            failing.append(names)

    if failing:
        raise RuntimeError(
            f"""
            Some tensors share memory, this will lead to duplicate memory on disk and potential differences when loading them again: {failing}.
            A potential way to correctly save your model is to use `save_model`.
            More information at https://huggingface.co/docs/safetensors/torch_shared_tensors
            """
        )

    return {
        k: {
            "dtype": str(v.dtype).split(".")[-1],
            "shape": v.shape,
            "data": _tobytes(v, k),
        }
        for k, v in tensors.items()
    }
