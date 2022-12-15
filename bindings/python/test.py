import datetime
from transformers import pipeline
import torch


print("============== DUMMY ====================")

start = datetime.datetime.now()
device = "cpu"
generator = pipeline("text-generation", model="gpt2", max_new_tokens=20, device=device, do_sample=False)
print(f"Loaded in {datetime.datetime.now() - start}")

out = generator("test")
print(out)
print(f"Ran in {datetime.datetime.now() - start}")

print("============== NO ALLOC ====================")
from accelerate.big_modeling import init_empty_weights

# from transformers.modeling_utils import no_init_weights
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

start = datetime.datetime.now()
filename = hf_hub_download("gpt2", filename="model.safetensors")
print(f"Lookup in {datetime.datetime.now() - start}")

with init_empty_weights():
    config = AutoConfig.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_config(config).eval()
    print(f"model from config in {datetime.datetime.now() - start}")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print(f"tokenizer in {datetime.datetime.now() - start}")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=20, do_sample=False)
    generator.device = torch.device(device)
    print(f"Loaded in {datetime.datetime.now() - start}")

weights = load_file(filename, device=device)
print(f"Loaded weights in {datetime.datetime.now() - start}")
model = generator.model
parameters = dict(model.named_parameters())

for name, tensor in weights.items():
    if name.endswith(".attn.bias"):
        continue
    full_name = f"transformer.{name}"
    module_name, param_name = full_name.rsplit(".", 1)
    module = model.get_submodule(module_name)
    current_tensor = parameters[full_name]
    module._parameters[param_name] = tensor
    if name == "wte.weight":
        model.lm_head._parameters["weight"] = tensor
model.to(device=device)

print(f"Loaded on model in {datetime.datetime.now() - start}")

out = generator("test")
print(f"Ran in {datetime.datetime.now() - start}")
print(out)
out = generator("test")
print(f"Ran in {datetime.datetime.now() - start}")
print(out)
