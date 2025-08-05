"""Simple utility tool to convert automatically most downloaded models"""

from convert import AlreadyExists, convert
from huggingface_hub import HfApi, ModelFilter, ModelSearchArguments
from transformers import AutoConfig


if __name__ == "__main__":
    api = HfApi()
    args = ModelSearchArguments()

    total = 50
    models = list(
        api.list_models(
            filter=ModelFilter(library=args.library.Transformers),
            sort="downloads",
            direction=-1,
        )
    )[:total]

    correct = 0
    errors = set()
    for model in models:
        model = api.model_info(model.id, files_metadata=True)
        size = None
        for sibling in model.siblings:
            if sibling.rfilename == "pytorch_model.bin":
                size = sibling.size
        if size is None or size > 2_000_000_000:
            print(f"[{model.downloads}] Skipping {model.modelId} (too large {size})")
            continue

        model_id = model.modelId
        print(f"[{model.downloads}] {model.modelId}")
        try:
            convert(api, model_id)
            correct += 1
        except AlreadyExists as e:
            correct += 1
            print(e)
        except Exception as e:
            config = AutoConfig.from_pretrained(model_id)
            errors.add(config.__class__.__name__)
            print(e)

    print(f"Errors: {errors}")
    print(f"File size is difference {len(errors)}")
    print(f"Correct rate {correct}/{total} ({correct / total * 100:.2f}%)")
