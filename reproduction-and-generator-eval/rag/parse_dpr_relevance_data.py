"""
This script reads DPR retriever training data and parses each datapoint. We save a line per datapoint.
Each line consists of the query followed by a tab-separated list of Wikipedia page titles constituting
positive contexts for a given query.
"""

import argparse
import json

from tqdm import tqdm


def parse_dpr_relevance_data(src_path, evaluation_set, gold_data_path):
    with open(src_path, "r") as src_file, open(evaluation_set, "w") as eval_file, open(
        gold_data_path, "w"
    ) as gold_file:
        dpr_records = json.load(src_file)
        for dpr_record in tqdm(dpr_records):
            eval_file.write(dpr_record["question"] + "\n")
            gold_file.write(dpr_record["question"] + "\t" + str(dpr_record["answers"]) + "\n")


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--src_path",
        type=str,
        default="biencoder-nq-dev.json",
        help="Path to raw DPR training data",
    )
    parser.add_argument(
        "--evaluation_set",
        type=str,
        help="where to store parsed evaluation_set file",
    )
    parser.add_argument(
        "--gold_data_path",
        type=str,
        help="where to store parsed gold_data_path file",
    )
    args = parser.parse_args()

    parse_dpr_relevance_data(args.src_path, args.evaluation_set, args.gold_data_path)


if __name__ == "__main__":
    main()
