import ast
import json
import os
import pandas as pd
import re
import time
import numpy as np
from loguru import logger

def update_aspect_count(aspect_dict, aspect):
    if aspect in aspect_dict:
        aspect_dict[aspect] += 1
    else:
        aspect_dict[aspect] = 1

def load_data(input_path):
    if input_path.endswith(".pkl"):
        data_df = pd.read_pickle(input_path)
    elif input_path.endswith("json") or input_path.endswith("jsonl"):
        data_df = pd.read_json(input_path, lines=True)
    elif input_path.endswith(".csv"):
        data_df = pd.read_csv(input_path)
    else:
        raise TypeError
    return data_df


def save_result(result, output_path):
    logger.debug(output_path)
    logger.debug(os.path.exists(os.path.dirname(output_path)))
    if output_path.endswith(".pkl"):
        result.to_pickle(output_path)
    elif output_path.endswith(".json") or output_path.endswith(".jsonl"):
        result.to_json(output_path, orient="records", lines=True)

def process_result(result_list, aspect_count):
    if isinstance(result_list, dict) and len(result_list) == 1 and "review" in result_list:
        result_list = result_list["review"]

    if isinstance(result_list, list):
        for r in result_list:
            if isinstance(r, dict):
                update_aspect_count(aspect_count, r["aspect"])
            else:
                logger.debug(f"Wrong format\n{r}")
    elif isinstance(result_list, dict):
        update_aspect_count(aspect_count, result_list["aspect"])

def string_to_list(string_to_change):
    logger.debug(string_to_change)
    list_string = re.findall(
        r"\[\s*(?:[^][]*|\[(?:[^][]*|\[(?:[^][]*)*\])*\])*\s*\]", string_to_change
    )
    logger.debug(list_string)
    if len(list_string) == 0 or list_string is None:
        logger.debug("Didn't find a valid Python list")
        potential_eval = string_to_change
    else:
        potential_eval = list_string[0]
        potential_eval = re.sub("\\n\s*", "", potential_eval)
    potential_list = ast.literal_eval(potential_eval)
    return potential_list


def build_review_context(review_list, context_length):
    all_reviews = "\n".join(review_list)
    all_context_reviews = all_reviews.split(" ")[:context_length]
    all_reviews_new = " ".join(all_context_reviews)
    all_reviews_new = ".".join(all_reviews_new.split(".")[:-1])
    logger.debug(
        f'The original review has length {len(all_reviews.split(" "))}\nThe new review has length{len(all_reviews_new.split(" "))}'
    )
    return all_reviews_new


def string_to_json(string_to_change):
    potential_list = []
    string_to_change = string_to_change.strip("```").strip("json")
    string_to_change = re.sub(r"(?<!\\)'s ", "\\'s ", string_to_change)
    # time_now = time.time()
    # list_string = None
    # while (time.time() - time_now < 5) and (list_string is None):
    #     list_string = re.findall(r"\[\s*(?:[^][]*|\[(?:[^][]*|\[(?:[^][]*)*\])*\])*\s*\]", string_to_change)
    # if len(list_string) == 0 or list_string is None:
    #     logger.debug("Didn't find a valid Python list")
    #     potential_eval = string_to_change
    # else:
    #     potential_eval = list_string[0]
    #     potential_eval = re.sub('\\n\s*', '', potential_eval)

    try:
        potential_list = ast.literal_eval(string_to_change)
    except SyntaxError as e:
        logger.debug(e)
        logger.debug(string_to_change)
        # exit()
        # if str(e).startswith('EOL') or str(e).startswith('unterminated string literal'):
        #     logger.debug(e)
        #     logger.debug(string_to_change)
        # else:
        #     string_to_change = string_to_change.replace('(?<!\\)\'s ', '\\\'s ')
        #     potential_list = ast.literal_eval(string_to_change)
    except ValueError as e1:
        logger.debug(e1)
        
    logger.debug(potential_list)
    logger.debug(type(potential_list))
    return potential_list


def choose_columns(data_path):
    dir_name = os.path.dirname(data_path)
    logger.debug(re.search(dir_name, "amasum"))
    if re.search("kialo", dir_name):
        return "text_to"
    elif re.search("cmv", dir_name):
        return "preprocessed_comment"
    elif re.search("Amasum", dir_name):
        return "text"
    elif re.search("SubjQA", dir_name):
        return "review"
    else:
        raise TypeError


def extract_key(protential_json, key):
    data = None
    try:
        protential_json = re.sub(
            r'(?<=\d)\s?("|\')(?=\s|,|}|$)', r"\"", protential_json
        )
    except TypeError as e:
        logger.debug(protential_json)
        logger.warning(e)
        exit()
    try:
        data = json.loads(protential_json)
    except json.decoder.JSONDecodeError as e:
        logger.warning(e)
        logger.warning(protential_json)
    if data and key in data:
        return data[key]
    else:
        return data


def split_list_in_reviews_into_sentences(sampled_comment_df):
    # Find
    min_list_size = 1
    mask = sampled_comment_df["sentences"].apply(
        lambda s: len(re.findall("\n[-+]", s)) >= min_list_size
    )

    # Process
    sampled_comment_df.loc[~mask, "sentences"] = sampled_comment_df.loc[
        ~mask, "sentences"
    ].apply(lambda x: [x])
    sampled_comment_df.loc[mask, "sentences"] = sampled_comment_df[mask][
        "sentences"
    ].apply(
        lambda s: [
            sent.strip("\n").strip()
            for sent in re.split("\n[\+\-]+", s)
            if len(sent.strip("\n").strip()) > 0 and sent.strip("\n").strip()[-1] != ":"
        ]
    )
    sampled_comment_df = sampled_comment_df.explode(["sentences"])
    return sampled_comment_df


def split_newlines_in_reviews_into_sentences(sampled_comment_df):
    # Find
    min_consecutive_newline = 2
    mask = sampled_comment_df["sentences"].apply(
        lambda s: re.match(
            "^.+\n{%d,}" % min_consecutive_newline, s.strip("\n").strip()
        )
        != None
    )

    # Process
    sampled_comment_df.loc[~mask, "sentences"] = sampled_comment_df.loc[
        ~mask, "sentences"
    ].apply(lambda x: [x])
    sampled_comment_df.loc[mask, "sentences"] = sampled_comment_df[mask][
        "sentences"
    ].apply(
        lambda s: [
            sent.strip("\n").strip()
            for sent in re.split("\n{%d,}" % min_consecutive_newline, s)
            if len(sent.strip("\n").strip()) > 0 and sent.strip("\n").strip()[-1] != ":"
        ]
    )
    sampled_comment_df = sampled_comment_df.explode(["sentences"])
    return sampled_comment_df


def preprocessing_comment_data(sampled_comment_df):
    preprocessed_sampled_comment_df = sampled_comment_df.copy()
    preprocessed_sampled_comment_df = split_list_in_reviews_into_sentences(
        preprocessed_sampled_comment_df
    )
    preprocessed_sampled_comment_df = split_newlines_in_reviews_into_sentences(
        preprocessed_sampled_comment_df
    )

    # Remove sent with less than 10 characters
    preprocessed_sampled_comment_df = preprocessed_sampled_comment_df[
        preprocessed_sampled_comment_df["sentences"].str.len() >= 10
    ]

    return preprocessed_sampled_comment_df
