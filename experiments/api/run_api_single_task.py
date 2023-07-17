import time
import json
from pathlib import Path

from fire import Fire
from tqdm import tqdm

from experiments.api.anthropic_api import AnthropicAPI
from experiments.api.openai_api import OpenAIAPI
from datasets import load_dataset

def generate_predictions_using_api(dataset_name: str, model_name: str = "text-davinci-003",
                                    log_progress_every_n_examples=20,
                                   limit_to_n_examples=None):
    model_folder_name = model_name.replace("-", "_")
    if model_name in ["text-davinci-003", "gpt-3.5-turbo", "gpt-4"]:
        api = OpenAIAPI(model_name, dataset_name)
    elif model_name in ["claude-v1","claude-v1.3"]:
        api = AnthropicAPI(model_name, dataset_name)
    else:
        raise ValueError(f"model_name {model_name} not supported")

    api.init_api()
    # load task data
    zero_scrolls_dataset = load_dataset("tau/zero_scrolls",dataset_name)["test"]
    predictions_file_path = Path(f"runs/api/{model_folder_name}/{dataset_name}/preds.jsonl")
    predictions_file_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"generating predictions for {dataset_name} with OpenAI {model_name}")

    #  API setup and parameters
    parameters = api.init_params()
    saved_predictions = 0
    predictions_file_path = predictions_file_path.parent / "preds.jsonl"
    with open(predictions_file_path, 'a') as f_out:

        for i, example in tqdm(enumerate(zero_scrolls_dataset)):
            if limit_to_n_examples is not None and i >= limit_to_n_examples:
                print(
                    f"Breaking when limit_to_n_examples is reached. i={i}, limit_to_n_examples={limit_to_n_examples}, newly generated predictions={saved_predictions}")
                break

            prompt = api.build_prompt(example)
            api.preprocess_parameters(parameters, prompt)

            time.sleep(0.5) # helps with rate limits
            response = api.call(parameters)
            output = api.build_output(example, prompt, parameters, response)

            # save output
            f_out.write(json.dumps(output) + "\n")
            saved_predictions += 1
            if i % log_progress_every_n_examples == 0:
                print(
                    f'saved {saved_predictions} predictions new predictions for {dataset_name} using {model_name}')

    print(
        f'finished generating {saved_predictions} predictions for {dataset_name} using OpenAI {model_name}')


if __name__ == '__main__':
    Fire(generate_predictions_using_api)
