import argparse
from collections import namedtuple
import pandas as pd
import logging

from verify_task import main as evaluate_dataset, DATASETS

log = logging.getLogger(__name__)

EXPECTED_DF_COLS = {"Task", "ID", "Prediction"}
EXPECTED_TASKS = [
    "gov_report",
    "summ_screen_fd",
    "qmsum",
    "narrative_qa",
    "qasper",
    "quality",
    "squality",
    "musique",
    "space_digest",
    "book_sum_sort"
]
assert set(EXPECTED_TASKS).issubset(DATASETS)
BenchmarkEvaluatorArgs = namedtuple(
    "BenchmarkEvaluatorArgs",
    "all_predictions split cache_dir output_dir internal_call",
)
DatasetEvaluatorArgs = namedtuple(
    "DatasetEvaluatorArgs",
    "predictions dataset_name split cache_dir output_dir internal_call",
)


def main(args):
    all_predictions = args.all_predictions
    if isinstance(all_predictions, str):
        all_predictions = load_predictions_df(all_predictions)
    errors = 0
    for task in EXPECTED_TASKS:

        log.info(f"Evaluating the results for task {task} with task {task}...")
        task_json = (
            all_predictions[all_predictions.Task == task][["ID", "Prediction"]]
            .set_index("ID")["Prediction"]
            .to_dict()
        )
        evaluator_obj = DatasetEvaluatorArgs(
            predictions=task_json,
            dataset_name=task,
            split=args.split,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            internal_call=True
        )
        try:
            evaluate_dataset(evaluator_obj, raise_on_errors=True)
        except Exception as e:
            errors += 1
            log.exception(f"Error for task: {task}:\n{e}")
            continue
        log.info(f"task: {task} is valid")
    if errors:
        msg = f"Found {errors} errors in the submission, see output files in {args.output_dir} for details."
        raise ValueError(msg)
    else:
        print("The verification was successful.")


def load_predictions_df(file_path):
    try:
        df = safe_read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read the csv with pandas: {e}")

    cols = set(df.columns)
    if cols != EXPECTED_DF_COLS:
        raise ValueError(f"csv file has invalid format. Expected columns {EXPECTED_DF_COLS} and got {cols} instead")

    tasks = set(df.Task.unique())
    if tasks != set(EXPECTED_TASKS):
        raise ValueError(
            f"csv file does not contain predictions for the expected tasks. "
            f"Expected tasks {sorted(EXPECTED_TASKS)} and got {sorted(tasks)} instead"
        )

    return df


def safe_read_csv(file_path):
    # https://stackoverflow.com/a/33952294
    return pd.read_csv(file_path, dtype=object, keep_default_na=False, na_values=["!@#$%^&*()"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the predictions for the full SCROLLS benchmark")
    parser.add_argument(
        "--all_predictions",
        type=str,
        help="Path to the file with all of the predictions or the actual predictions",
        required=True,
    )
    parser.add_argument("--output_dir", type=str, help="Directory of the output metrics file", required=True)
    parser.add_argument("--split", type=str, help="The split to evaluate on", default="test")
    parser.add_argument("--internal_call", type=str, help="For internal use", default=False)
    parser.add_argument(
        "--cache_dir", type=str, help="Cache dir for the dataset download", default=None, required=False
    )
    args = parser.parse_args()

    main(args)
