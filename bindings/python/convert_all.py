"""Simple utility tool to convert automatically most downloaded models"""
from huggingface_hub import HfApi, ModelFilter, ModelSearchArguments
from convert import convert


if __name__ == "__main__":
    api = HfApi()
    args = ModelSearchArguments()

    total = 100
    models = list(api.list_models(filter=ModelFilter(library=args.library.Transformers), sort="downloads", direction=-1))[:total]

    correct = 0
    for model in models:
        print(f"[{model.downloads}] {model.modelId}")
        try:
            result = convert(api, model.modelId)
            if result is not None:
                correct += 1
        except ValueError as e:
            print(e)

    print(f"Correct rate {correct}/{total} ({correct/total * 100:.2f}%)")
