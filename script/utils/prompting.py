import json
import re
from tqdm import tqdm
from os import listdir
import openai
import os
import pandas as pd
from openai import OpenAI
from pathlib import Path
import random
import re
from tqdm import tqdm
from loguru import logger
import yaml

from utils.preprocessing import string_to_list

# from mistral_common.protocol.instruct.messages import UserMessage
# from mistral_common.protocol.instruct.request import ChatCompletionRequest
# from mistral_inference.generate import generate

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "../../prompt")


def load_prompt_template(file_path, ptype):
    if file_path.endswith(".txt"):
        with open(file_path) as f:
            sys_prompt = "".join([line for line in f])
    elif file_path.endswith(".yaml"):
        with open(file_path) as f:
            prompt_dict = yaml.safe_load(f)
            if ptype in prompt_dict:
                sys_prompt = prompt_dict[ptype]
            else:
                raise NotImplementedError
    return sys_prompt


def get_prompt(name, ptype=None, aspect_set=None):
    sys_prompt = ""
    if name.endswith(".txt"):
        with open(os.path.join(PROMPT_DIR, name)) as f:
            sys_prompt = "".join([line for line in f])
    elif name.endswith(".yaml"):
        with open(os.path.join(PROMPT_DIR, name), "r") as f:
            prompt_dict = yaml.safe_load(f)
            if ptype in prompt_dict:
                sys_prompt = prompt_dict[ptype]
            else:
                raise NotImplementedError
    if aspect_set:
        # logger.debug(sys_prompt)
        sys_prompt = re.sub("\{aspect\}", aspect_set, sys_prompt)
        # logger.debug(sys_prompt)
    return sys_prompt


def get_completion(model, prompt, max_tokens, temperature=0):
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].text


def get_chat_completion(
    model, prompt, max_tokens, temperature=0, tokenizer=None, model_obj=None
):
    # messages = [{"role": "user", "content": prompt}]
    result = "This is the initial message"
    if model.startswith("gpt"):
        client = OpenAI()
        # if 'json' in prompt:
        response = client.chat.completions.create(
            model=model,
            messages=prompt,
            max_tokens=max_tokens,
            temperature=temperature,  # this is the degree of randomness of the model's output
            response_format=(
                {"type": "json_object"} if "json" in str(prompt[-1]).lower() else None
            ),
        )
        result = response.choices[0].message.content
        result = result.strip("```").strip("json")

    elif model.startswith("qwen") and tokenizer is not None and model_obj is not None:
        model_input = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        model_input = tokenizer([model_input], return_tensors="pt").to(model_obj.device)
        generate_ids = model_obj.generate(**model_input, max_new_tokens=max_tokens)
        generate_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_input.input_ids, generate_ids)
        ]

        result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
        logger.debug(f"Here is the output of {model}\n{result}")

    elif (
        model.startswith("mistral") and tokenizer is not None and model_obj is not None
    ):
        logger.debug(prompt)
        completion_request = ChatCompletionRequest(
            messages=[UserMessage(content=prompt)]
        )
        tokens = tokenizer.encode_chat_completion(completion_request).tokens

        out_tokens, _ = generate(
            [tokens],
            model_obj,
            max_tokens=max_tokens,
            temperature=0.0,
            eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
        )
        result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
        result = result.strip("Anwser:")
        logger.debug(result)
    elif model.startswith("gemma") and tokenizer is not None and model_obj is not None:
        input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
        logger.debug("here")
        outputs = model_obj.generate(**input_ids, max_new_tokens=max_tokens)
        logger.debug("Generate success")
        result = tokenizer.decode(outputs[0])
        logger.debug(result)
    return result


fix_prompt = """Fix the input text, delimited by triple quote, into the correct JSON format.
JSON input: [{'aspect': 'wraps','salads', 'burgers','mango margarita','sentiment': 'positive'}]
Corrected JSON: [{'aspect': 'wraps', 'sentiment': 'positive'}, {'aspect': 'salads', 'sentiment': 'positive'}, {'aspect': 'burgers', 'sentiment': 'positive'}, {'aspect': 'mango margarita', 'sentiment': 'positive'}]

JSON input: %s
"""


def get_fix_completion(comment, offset=1, model="vicuna-7b-v1.3"):
    max_tokens = int(len(comment) * 1.2)
    response = openai.Completion.create(
        model=model,
        prompt=fix_prompt % comment,
        max_tokens=max(50, max_tokens) * offset,
        temperature=0,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].text


def fix_json(comment):
    offset = 1
    answers = []
    prev_response = ""
    while len(answers) == 0:
        response = get_fix_completion(comment, offset=offset)
        response = response.replace("\_", "_")
        answers = re.findall("Corrected JSON: (\[.*\])", response)
        offset += 1
        if response == prev_response or offset >= 5:
            answers = ["[]"]
            break
        prev_response = response

    corrected_json = answers[0]

    return corrected_json
