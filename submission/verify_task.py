import os
import argparse
import json

from datasets import load_dataset

DATASETS = [
    "narrative_qa",
    "qasper",
    "summ_screen_fd",
    "gov_report",
    "qmsum",
    "quality",
    "squality",
    "musique",
    "space_digest",
    "book_sum_sort",
]


def main(args, raise_on_errors=False):
    """
    If raise_on_errors is True, raises ValueError on verification errors (after dumping the error descriptions).
    Otherwise, exists with an error code
    """
    predictions = args.predictions
    dataset_name = args.dataset_name

    # Downloading and loading the dataset from the hub
    load_dataset_kwargs = {
        "path": "tau/zero_scrolls",
        "name": dataset_name,
    }
    if args.cache_dir is not None:
        load_dataset_kwargs["cache_dir"] = args.cache_dir
    load_dataset_kwargs["split"] = "test"
    seq2seq_dataset = load_dataset(**load_dataset_kwargs)

    # Prepare reference
    untokenized_dataset = drop_duplicates_in_input(seq2seq_dataset)
    id_to_labels = {instance["id"]: instance["outputs"] for instance in untokenized_dataset}

    # Prepare predictions
    if isinstance(predictions, str):
        with open(predictions) as f:
            id_to_pred = json.load(f)
    else:
        id_to_pred = predictions

    # Check for format errors
    errors, details = verify(id_to_pred, id_to_labels)

    out_file_path = get_errors_filename(args.output_dir, dataset_name)
    os.makedirs(args.output_dir, exist_ok=True)

    if len(errors) > 0:
        # Output errors
        errors_msg = errors[0] if len(errors) == 1 else " ".join(f"{i}: {err}" for i, err in enumerate(errors))
        print(json.dumps(errors, indent=4))
        print(f"See details in: {out_file_path}")
        with open(out_file_path, mode="w") as f:
            json.dump({"errors": errors, "details": details}, f, indent=4)
        if raise_on_errors:
            raise ValueError(f"Failed to evaluate due to: {errors_msg}")
        exit(os.EX_DATAERR)


# Copied from baselines/src/utils/duplicates.py
def drop_duplicates_in_input(untokenized_dataset):
    indices_to_keep = []
    id_to_idx = {}
    outputs = []
    for i, (id_, output) in enumerate(zip(untokenized_dataset["id"], untokenized_dataset["output"])):
        if id_ in id_to_idx:
            outputs[id_to_idx[id_]].append(output)
            continue
        indices_to_keep.append(i)
        id_to_idx[id_] = len(outputs)
        outputs.append([output])
    untokenized_dataset = untokenized_dataset.select(indices_to_keep).flatten_indices()
    untokenized_dataset = untokenized_dataset.remove_columns("output")
    untokenized_dataset = untokenized_dataset.add_column("outputs", outputs)
    return untokenized_dataset


def get_errors_filename(outdir, dataset_name):
    return os.path.join(outdir, f"{dataset_name}_errors.json")


def verify(id_to_pred, id_to_labels):
    errors = []
    details = {"missing_keys": [], "redundant_keys": []}
    if not isinstance(id_to_pred, dict):
        errors.append('The predictions must be saved a JSON object: {"id1": "prediction1", "id2": "prediction2", ...}')
    else:
        if not all(isinstance(key, str) for key in id_to_pred.keys()):
            errors.append("All keys of the predictions dictionary must be strings")
        if not all(isinstance(value, str) for value in id_to_pred.values()):
            errors.append("All values of the predictions dictionary must be strings")
        if len(errors) == 0:
            predictions_keys, reference_keys = set(id_to_pred.keys()), set(id_to_labels.keys())
            missing_keys = reference_keys - predictions_keys
            redundant_keys = predictions_keys - reference_keys

            if len(missing_keys) > 0:
                details["missing_keys"] = list(missing_keys)
                errors.append(f"There are missing example IDs.")
            else:
                del details["missing_keys"]

            if len(redundant_keys) > 0:
                details["redundant_keys"] = list(redundant_keys)
                errors.append(f"There are redundant example IDs.")
            else:
                del details["redundant_keys"]

    return errors, details


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="verify ZeroSCROLLS predictions per dataset")
    parser.add_argument(
        "--predictions", type=str, help="Path to the predictions file or the actual predictions", required=True
    )
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset", choices=DATASETS, required=True)
    parser.add_argument("--output_dir", type=str, help="Directory of the output file", required=True)
    parser.add_argument("--internal_call", type=str, help="For internal use", default=False)
    parser.add_argument(
        "--cache_dir", type=str, help="Cache dir for the dataset download", default=None, required=False
    )
    args = parser.parse_args()

    main(args)
