"""Simple utility tool to convert automatically most downloaded models"""
from huggingface_hub import HfApi, ModelFilter, ModelSearchArguments
from convert import convert


if __name__ == "__main__":
    api = HfApi()
    args = ModelSearchArguments()

    total = 100
    models = list(api.list_models(filter=ModelFilter(library=args.library.Transformers), sort="downloads", direction=-1))[:total]

    correct = 0
    file_diff = set()
    not_contiguous = set()
    for model in models:
        model_id = model.modelId
        print(f"[{model.downloads}] {model.modelId}")
        try:
            result = convert(api, model_id)
            if result is not None:
                correct += 1
        except RuntimeError as e:
            file_diff.add( model_id)
            print(e)
        except ValueError as e:
            not_contiguous.add(model_id)
            print(e)

    print(f"Not contiguous: {not_contiguous}")
    print(f"File size is difference {file_diff}")
    print(f"Not contiguous: {len(not_contiguous)}")
    print(f"File size is difference {len(file_diff)}")
    print(f"Correct rate {correct}/{total} ({correct/total * 100:.2f}%)")
