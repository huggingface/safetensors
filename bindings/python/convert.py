import argparse
import json
import os
import shutil

import torch

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.file_download import repo_folder_name
from transformers import AutoConfig
from transformers.pipelines.base import infer_framework_load_model
from safetensors.torch import save_file


def check_file_size(sf_filename, pt_filename):
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size

    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError(f"""The file size different is more than 1%:
         - {sf_filename}: {sf_size}
         - {pt_filename}: {pt_size}
         """)


def rename(pt_filename) -> str:
    local = pt_filename.replace(".bin", ".safetensors")
    local = local.replace("pytorch_model", "model")
    return local


def convert_multi(model_id):
    filename = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin.index.json")
    with open(filename, "r") as f:
        data = json.load(f)

    filenames = set(data["weight_map"].values())
    for filename in filenames:
        cached_filename = hf_hub_download(repo_id=model_id, filename=filename)
        loaded = torch.load(cached_filename)
        sf_filename = rename(filename)

        local = os.path.join(folder, sf_filename)
        save_file(loaded, local, metadata={"format": "pt"})
        check_file_size(local, cached_filename)
        local_filenames.append(local)

    index = os.path.join(folder, "model.safetensors.index.json")
    with open(index, "w") as f:
        newdata = {k: v for k, v in data.items()}
        newmap = {k: rename(v) for k, v in data["weight_map"].items()}
        newdata["weight_map"] = newmap
        json.dump(newdata, f)
    local_filenames.append(index)

    operations = [CommitOperationAdd(path_in_repo=local.split("/")[-1], path_or_fileobj=local) for local in local_filenames]

    return operations


def convert_single(model_id, folder):
    sf_filename = "model.safetensors"
    filename = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
    loaded = torch.load(filename)

    local = os.path.join(folder, sf_filename)
    save_file(loaded, local, metadata={"format": "pt"})

    check_file_size(local, filename)

    operations = [CommitOperationAdd(path_in_repo=sf_filename, path_or_fileobj=local)]
    return operations

def check_final_model(model_id, folder):
    config = hf_hub_download(repo_id=model_id, filename="config.json")
    shutil.copy(config, os.path.join(folder, "config.json"))
    config = AutoConfig.from_pretrained(folder)
    _, sf_model = infer_framework_load_model(folder, config)
    _, pt_model = infer_framework_load_model(model_id, config)

    input_ids = torch.arange(10).long().unsqueeze(0)
    sf_logits = sf_model(input_ids)
    pt_logits = pt_model(input_ids)
    torch.testing.assert_close(sf_logits, pt_logits)
    print(f"Model {model_id} is ok !")


def convert(api, model_id):
    info = api.model_info(model_id)
    filenames = set(s.rfilename for s in info.siblings)

    folder = repo_folder_name(repo_id=model_id, repo_type="models")
    os.makedirs(folder)
    try:
        operations = None
        if "model.safetensors" in filenames or "model_index.safetensors.index.json" in filenames:
            print(f"Model {model_id} is already converted, skipping..")
        elif "pytorch_model.bin" in filenames:
            operations = convert_single(model_id, folder)
        elif "pytorch_model.bin.index.json" in filenames:
            operations = convert_multi(model_id, folder)
        else:
            raise RuntimeError(f"Model {model_id} doesn't seem to be a valid pytorch model. Cannot convert")

        if operations:
            check_final_model(model_id, folder)
            api.create_commit(
                repo_id=model_id,
                operations=operations,
                commit_message="Adding `safetensors` variant of this model",
                create_pr=True,
            )
    finally:
        shutil.rmtree(folder)
    return 1


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
    convert(api, model_id)
