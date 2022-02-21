from huggingface_hub import hf_hub_download
from transformers import AutoModel
import torch
import os
import datetime


def bench_auto_trained(model_id: str):
    AutoModel.from_pretrained(model_id)


def bench_torch_load(model_id: str):
    filename = hf_hub_download(model_id, filename="pytorch_model.bin")
    data = torch.load(filename)
    import ipdb

    ipdb.set_trace()


benches = [
    bench_torch_load,
    bench_auto_trained,
]


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def main():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for model_id in ["gpt2", "gpt2-medium"]:
        # Make sure everything is on disk
        benches[-1](model_id)
        filename = hf_hub_download(model_id, filename="pytorch_model.bin")
        size = os.path.getsize(filename)

        start = datetime.datetime.now()
        open(filename, "rb").read()
        print(
            f"{model_id} ({sizeof_fmt(size)}): pure read: {datetime.datetime.now() - start}"
        )
        for bench in benches:
            start = datetime.datetime.now()
            bench(model_id)
            print(
                f"{model_id} ({sizeof_fmt(size)}): {bench.__name__}: {datetime.datetime.now() - start}"
            )


if __name__ == "__main__":
    main()
