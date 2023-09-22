import argparse
import os

import numpy as np
import pandas as pd
import json

SUBMISSION_LINK = "https://zero.scrolls-benchmark.com/submission"
TASKS_MAPPING = {
    "gov_report_file": "gov_report",
    "summ_screen_fd_file": "summ_screen_fd",
    "qmsum_file": "qmsum",
    "squality_file": "squality",
    "qasper_file": "qasper",
    "narrative_qa_file": "narrative_qa",
    "quality_file": "quality",
    "musique_file": "musique",
    "space_digest_file": "space_digest",
    "book_sum_sort_file": "book_sum_sort",
}
COLUMNS = ["Task", "ID", "Prediction"]


def safe_read_csv(file_path):
    # https://stackoverflow.com/a/33952294
    return pd.read_csv(file_path, dtype=object, keep_default_na=False, na_values=["!@#$%^&*()"])


def main():
    parser = argparse.ArgumentParser(description="Prepare ZeroSCROLLS prediction")
    parser.add_argument("--output_dir", type=str, help="Path to output the prediction file", required=True)
    parser.add_argument(
        "--qmsum_file", type=str, help="The path to the qmsum dataset json file containing prediction", required=True
    )
    parser.add_argument(
        "--qasper_file",
        type=str,
        help="The path to the qasper dataset json file containing prediction",
        required=True,
    )
    parser.add_argument(
        "--summ_screen_fd_file",
        type=str,
        help="The path to the summ_screen dataset json file containing prediction",
        required=True,
    )
    parser.add_argument(
        "--quality_file",
        type=str,
        help="The path to the quality dataset json file containing prediction",
        required=True,
    )
    parser.add_argument(
        "--narrative_qa_file",
        type=str,
        help="The path to the narrative_qa dataset json file containing prediction",
        required=True,
    )
    parser.add_argument(
        "--gov_report_file",
        type=str,
        help="The path to the gov_report dataset json file containing prediction",
        required=True,
    )
    parser.add_argument(
        "--squality_file",
        type=str,
        help="The path to the squality dataset json file containing prediction",
        required=True,
    )
    parser.add_argument(
        "--musique_file",
        type=str,
        help="The path to the musique dataset json file containing prediction",
        required=True,
    )
    parser.add_argument(
        "--space_digest_file",
        type=str,
        help="The path to the space_digest dataset json file containing prediction",
        required=True,
    )
    parser.add_argument(
        "--book_sum_sort_file",
        type=str,
        help="The path to the book_sum_sort dataset json file containing prediction",
        required=True,
    )
    args = parser.parse_args()

    tasks_dfs = pd.DataFrame(columns=COLUMNS, data=[])
    for file_key, task_name in TASKS_MAPPING.items():
        print(f"Adding prediction for {task_name} from {file_key}...")
        with open(getattr(args, file_key)) as f:
            task_data = json.load(f)
        task_df = pd.DataFrame.from_dict(task_data, orient="index", columns=COLUMNS[-1:]).reset_index(drop=False)
        task_df[COLUMNS[0]] = task_name
        task_df[COLUMNS[1]] = task_df["index"]
        tasks_dfs = pd.concat((tasks_dfs, task_df[COLUMNS]))

    os.makedirs(args.output_dir, exist_ok=True)
    outfile = os.path.join(args.output_dir, "zero_scrolls_predictions.csv")
    print(f"Saving the complete prediction file to: {outfile}")
    tasks_dfs = tasks_dfs.reset_index(drop=True)
    tasks_dfs.to_csv(outfile, index=False)

    print("validating submission file is exactly the same as expected")
    recovered_tasks_dfs = safe_read_csv(outfile)
    assert len(recovered_tasks_dfs) == len(tasks_dfs)
    assert recovered_tasks_dfs.columns.tolist() == tasks_dfs.columns.tolist()
    assert np.all(recovered_tasks_dfs.values == tasks_dfs.values)

    print(f"Your benchmark prediction file is ready. If it contains prediction for the test sets please head over to {SUBMISSION_LINK} to submit to the ZeroSCROLLS leaderboard.")


if __name__ == "__main__":
    main()
