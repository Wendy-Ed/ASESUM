from argparse import ArgumentParser
import glob
import json
import numpy as np
import os
import pandas as pd
import re
from loguru import logger
import yaml

# from transformers import AutoTokenizer, AutoModelForCausalLM
from extractor import AspectExtractor
from prompter import QwenPrompter, GPTPrompter
from utils.preprocessing import save_result



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="../data/kialo/kialo-pta24.pkl",
        required=True,
        help="The directory has the arguments",
    )
    parser.add_argument("--config", type=str, default="config/aspect_extract.yaml")
    parser.add_argument("--seed_aspects", action="store_true")
    parser.add_argument("--output_path", type=str, default=None, required=True)

    args = parser.parse_args()
    tokenizer, model = None, None
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if config['model_name'].startswith("qwen"):
        prompter = QwenPrompter(config['model_name'], config)
    elif config['model_name'].startswith("gpt"):
        prompter = GPTPrompter(config['model_name'], config)
    else:
        raise NotImplementedError

    logger.debug("Finish loading models")
    logger.debug("Extracting the aspect")
    extractor = AspectExtractor(
        config["prompt_path"], config["prompt_type"], prompter=prompter
    )

    for file in glob.glob(os.path.join(args.input_path, f"*.json")):
        
        if os.path.exists(os.path.join(args.input_path, "test_result", os.path.basename(file).split(".")[0] + f'_{config["model_name"]}_TEST.json')):
            logger.debug("This file has been extracted....Continue")
            continue

        logger.debug("===" * 20)
        logger.debug(f"File: {os.path.basename(file)}")
        with open(file) as fl:
            data = json.load(fl)
            data_df = pd.DataFrame(data['customer_reviews'])
            product_id = os.path.basename(file).split(".")[0]
            logger.debug(f"This product {product_id} has {data_df.shape[0]} reviews.")
            list_starting_aspect = extractor.extract(data, **{"column": "product_meta"})
            logger.debug(list_starting_aspect)


        logger.debug(f"Finish loading dataset, the size is {data_df.shape}")
        extractor.extract(data_df, **{"column": config["extract_column"], "category": list_starting_aspect})
        save_result(
            data_df,
            os.path.join(
                args.output_path,
                os.path.basename(file).strip(".json")
                + f"_{config['extract_column']}"
                + f"_{config['model_name']}"
                + "_TEXT.json",
            ),
        )
    logger.debug(data_df.columns)

    logger.debug("Saved!")
