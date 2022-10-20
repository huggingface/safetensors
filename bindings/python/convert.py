import argparse
import json
import os

import torch

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from safetensors.torch import save_file


def rename(pt_filename) -> str:
    local = pt_filename.replace(".bin", ".safetensors")
    local = local.replace("pytorch_model", "model")
    return local


def convert_multi(model_id):
    local_filenames = []
    try:
        filename = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin.index.json")
        with open(filename, "r") as f:
            data = json.load(f)

        filenames = set(data["weight_map"].values())
        for filename in filenames:
            cached_filename = hf_hub_download(repo_id=model_id, filename=filename)
            loaded = torch.load(cached_filename)
            local = rename(filename)
            save_file(loaded, local, metadata={"format": "pt"})
            local_filenames.append(local)

        index = "model.safetensors.index.json"
        with open(index, "w") as f:
            newdata = {k: v for k, v in data.items()}
            newmap = {k: rename(v) for k, v in data["weight_map"].items()}
            newdata["weight_map"] = newmap
            json.dump(newdata, f)
        local_filenames.append(index)

        api = HfApi()
        operations = [CommitOperationAdd(path_in_repo=local, path_or_fileobj=local) for local in local_filenames]
        api.create_commit(
            repo_id=model_id,
            operations=operations,
            commit_message="Adding `safetensors` variant of this model",
            create_pr=True,
        )
    finally:
        for local in local_filenames:
            os.remove(local)


def convert_single(model_id):
    local = "model.safetensors"
    try:
        filename = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
        loaded = torch.load(filename)
        save_file(loaded, local, metadata={"format": "pt"})

        api = HfApi()

        api.upload_file(
            path_or_fileobj=local,
            create_pr=True,
            path_in_repo=local,
            repo_id=model_id,
        )
    finally:
        os.remove(local)


if __name__ == "__main__":
    DESCRIPTION = """
    Simple utility tool to convert automatically some weights on the hub to `safetensors` format.
    It is PyTorch exclusive for now.
    It works by downloading the weights (PT), converting them locally, and uploading them back
    as a PR on the hub.
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "model_id",
        type=str,
        help="The name of the model on the hub to convert. E.g. `gpt2` or `facebook/wav2vec2-base-960h`",
    )
    args = parser.parse_args()
    model_id = args.model_id
    api = HfApi()
    info = api.model_info(model_id)
    filenames = set(s.rfilename for s in info.siblings)
    if "pytorch_model.bin" in filenames:
        convert_single(model_id)
    else:
        convert_multi(model_id)
