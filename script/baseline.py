import os
import glob
import json
import yaml
from argparse import ArgumentParser

import pandas as pd
from loguru import logger

from extractor import BaselineExtrator
from prompter import GPTPrompter, QwenPrompter

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="../data",
        required=False,
        help="The directory has the arguments",
    )
    parser.add_argument("--config", type=str, default="config/baseline.yaml")
    parser.add_argument("--context_length", type=int, default=4096)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--output_path", type=str, default=None, required=True)

    args = parser.parse_args()
    tokenizer, model = None, None
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        if config['model_name'].startswith("qwen"):
            prompter = QwenPrompter(config['model_name'], config)
        elif config['model_name'].startswith("gpt"):
            prompter = GPTPrompter(config["model_name"], config)
        else:
            raise NotImplementedError
    summarizer = BaselineExtrator(
        config["prompt_path"], config["prompt_type"], prompter=prompter
    )
    logger.debug("Finish loading models")

    summaries = []
    for file in glob.glob(os.path.join(args.input_path, f"*.json")):
        logger.debug("===" * 20)
        logger.debug(f"File: {os.path.basename(file)}")
        # data_df = sample_input_data(file)
        with open(file) as fl:
            data_list = json.load(fl)
            data_df = pd.DataFrame(data_list["customer_reviews"])
        logger.debug(f"Looking for {os.path.join(args.output_path, file)}")

        if os.path.exists(
            os.path.join(
                args.output_path,
                os.path.basename(file).strip(".json")
                + f"_{config['extract_column']}"
                + f"_{config['model_name']}"
                + "_btest.json",
            ),
        ):
            logger.debug("This file has already been extracted...")
            continue
        else:

            product_id = os.path.basename(file).strip(".json")
            # reviews = build_review_context(
            #     data_df["text"].values.tolist(), args.context_length - data_df.shape[0]
            # )
            data_list["entity_id"] = product_id
            logger.debug(f"This product is {product_id}")

            summary = summarizer.extract(
                data_df,
                config["model_name"],
                **{"context_length": args.context_length, "column": config['extract_column']},
            )
            logger.debug(f"Generated summary is:\n{summary}")
            data_list["baseline_summary"] = [summary]
            if args.batch:
                summaries.append(data_list)
            else:
                with open(
                    os.path.join(
                        args.output_path,
                        os.path.basename(file).strip(".json")
                        + f"_{config['extract_column']}"
                        + f"_{config['model_name']}"
                        + "_btest.json",
                    ),
                    "w",
                ) as f1:
                    f1.write(json.dumps(data_list, indent=4))
    if args.batch:
        with open(
            os.path.join(
                args.output_path,
                f"{config['model_name']}"
                + f"_{config['extract_column']}"
                + "_btest.json",
            ),
            "w",
        ) as f1:
            for s in summaries:
                f1.write(json.dumps(s) + "\n")
    logger.debug("Saved!")
