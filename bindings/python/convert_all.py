"""Simple utility tool to convert automatically most downloaded models"""
from huggingface_hub import HfApi, ModelFilter, ModelSearchArguments
from convert import convert


if __name__ == "__main__":
    api = HfApi()
    args = ModelSearchArguments()

    total = 100
    models = list(api.list_models(filter=ModelFilter(library=args.library.Transformers), sort="downloads", direction=-1))[:total]

    correct = 0
    errors = set()
    for model in models:
        model_id = model.modelId
        print(f"[{model.downloads}] {model.modelId}")
        try:
            result = convert(api, model_id)
            if result is not None:
                correct += 1
        except Exception as e:
            errors.add( model_id)
            print(e)


    print(f"Errors: {errors}")
    print(f"File size is difference {len(errors)}")
    print(f"Correct rate {correct}/{total} ({correct/total * 100:.2f}%)")
