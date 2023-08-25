# Convert weights to safetensors

PyTorch model weights are commonly saved and stored as `.bin` files with Python's [`pickle`](https://docs.python.org/3/library/pickle.html) utility. To save and store your model weights in the more secure `safetensor` format, we recommend converting your weights to `.safetensors`.

The easiest way to convert your model weights is to use the [Convert Space](https://huggingface.co/spaces/diffusers/convert), given your model weights are already stored on the Hub. The Convert Space downloads the pickled weights, converts them, and opens a Pull Request to upload the newly converted `.safetensors` file to your repository. Merge the Pull Request to upload the weights, but if you can't wait to try it out, you can also use the `.safetensors` immediately by specifying the reference to the Pull Request in the revision parameter:

```py
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "my-safe-model", revision="refs/pr/1", use_safetensors=True
)
```

Another way to convert your `.bin` files is to use the [`~safetensors.torch.save_model`] function:

```py
from transformers import AutoModel
from safetensors.torch import save_model

unsafe_model = AutoModel.from_pretrained("my-unsafe-model")
save_model(unsafe_model, "model.safetensors")
```
