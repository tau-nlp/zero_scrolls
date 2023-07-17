import os
import sys

from fire import Fire


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.hf.run_hf_model import datasets
from experiments.api.run_api_single_task import generate_predictions_using_api


def main(model_name: str, limit_to_n_examples: int = None):
    for dataset in datasets:
        print(f"Starting with {dataset}")
        generate_predictions_using_api(dataset_name=dataset, model_name=model_name,
                                       limit_to_n_examples=limit_to_n_examples)


if __name__ == '__main__':
    Fire(main)
