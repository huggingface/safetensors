"""Test multi-device sharded loading with device_map."""
import sys
import json
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open
import torch

def main():
    model_id = "Qwen/Qwen2.5-14B"
    cache_dir = "/raid/hf_cache"

    print(f"Resolving model: {model_id}")
    model_path = snapshot_download(
        model_id, cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json"],
        ignore_patterns=["*.bin", "*.pt", "*.onnx"],
    )

    index_file = Path(model_path) / "model.safetensors.index.json"
    with open(index_file) as f:
        idx = json.load(f)

    weight_map = idx["weight_map"]
    tensor_names = list(weight_map.keys())
    print(f"Model: {model_path}")
    print(f"Total tensors: {len(tensor_names)}")

    # Build a simple device_map: split tensors across 2 GPUs
    device_map = {}
    for i, name in enumerate(tensor_names):
        device_map[name] = f"cuda:{i % 2}"

    devices_used = set(device_map.values())
    per_device = {d: sum(1 for v in device_map.values() if v == d) for d in devices_used}
    print(f"Device map: {per_device}")

    # Initialize CUDA on both devices before testing
    torch.cuda.init()
    _ = torch.zeros(1, device="cuda:0")
    _ = torch.zeros(1, device="cuda:1")
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    print("CUDA initialized on devices 0 and 1")

    # Test 1: Load with device_map using cuFile backend
    for backend in ["cufile", "io_uring"]:
        print(f"\n{'='*60}")
        print(f"Testing backend={backend} with multi-device...")
        print(f"{'='*60}")

        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        t0 = time.perf_counter()

        try:
            with safe_open(
                str(index_file),
                framework="pt",
                device=device_map,
                backend=backend,
            ) as sf:
                t_open = time.perf_counter() - t0
                print(f"  safe_open took: {t_open:.3f}s")

                keys = sf.keys()
                print(f"  keys(): {len(keys)} tensors")

                # Verify a few tensors from each device
                checked = {0: 0, 1: 0}
                errors = []
                for name in tensor_names[:20]:
                    try:
                        t = sf.get_tensor(name)
                        expected_dev = int(device_map[name].split(":")[1])
                        actual_dev = t.device.index
                        if actual_dev != expected_dev:
                            errors.append(f"  {name}: expected cuda:{expected_dev}, got cuda:{actual_dev}")
                        else:
                            checked[expected_dev] += 1
                    except Exception as e:
                        errors.append(f"  {name}: {e}")

                if errors:
                    print(f"  ERRORS:")
                    for e in errors:
                        print(f"    {e}")
                else:
                    print(f"  First 20 tensors verified OK: {checked}")

                # Full load timing
                torch.cuda.synchronize(0)
                torch.cuda.synchronize(1)
                t1 = time.perf_counter()
                for name in tensor_names:
                    t = sf.get_tensor(name)
                torch.cuda.synchronize(0)
                torch.cuda.synchronize(1)
                t_load = time.perf_counter() - t1
                print(f"  All tensors loaded in: {t_load:.3f}s")

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Test 2: Compare with single-device loading for reference
    print(f"\n{'='*60}")
    print(f"Reference: single-device cuFile (cuda:0)")
    print(f"{'='*60}")

    torch.cuda.synchronize(0)
    t0 = time.perf_counter()
    with safe_open(
        str(index_file),
        framework="pt",
        device="cuda:0",
        backend="cufile",
    ) as sf:
        t_open = time.perf_counter() - t0
        print(f"  safe_open took: {t_open:.3f}s")

        torch.cuda.synchronize(0)
        t1 = time.perf_counter()
        for name in tensor_names:
            t = sf.get_tensor(name)
        torch.cuda.synchronize(0)
        t_load = time.perf_counter() - t1
        print(f"  All tensors loaded in: {t_load:.3f}s")

    print("\nDone!")


if __name__ == "__main__":
    main()
