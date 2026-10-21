import anthropic
import os
import openai
import json
import requests
import torch
import time


from openai import OpenAI
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.prompting import load_prompt_template


PROMPT_DIR = os.path.join(os.path.dirname(__file__), "../prompt")


class Prompter:
    def __init__(self, model):
        self.model = model
        pass

    def build_sys_prompt(self, prompt_file, prompt_type, parameter_dict=None):
        """Get the system prompt and fill in the possible parameters"""
        prompt_template = load_prompt_template(
            os.path.join(PROMPT_DIR, prompt_file), prompt_type
        )
        if parameter_dict is not None:
            logger.debug(parameter_dict)
            logger.debug(prompt_template)
            prompt = prompt_template.format(**parameter_dict)
        else:
            prompt = prompt_template

        # logger.debug(f"Here is the prompt\n{prompt}")
        return prompt

    def build_openai_messages(
        self, user_input, prompt_file, prompt_type, parameters, message_type
    ):
        logger.debug(user_input)
        message = []
        if message_type == "system":
            message.append(
                {
                    "role": "system",
                    "content": self.build_sys_prompt(
                        prompt_file, prompt_type, parameters
                    ),
                }
            )
        message.append({"role": "user", "content": user_input})
        return message

    def prompting(self, user_input, sys_prompt):
        raise NotImplemented


class GPTPrompter(Prompter):
    def __init__(self, model, config):
        super().__init__(model)
        self.temperature = 0
        self.max_token_ratio = 10
        self.client = OpenAI()

    def price_calculation(self, response):
        price_table = {"gpt-5": {"input": 1.25, "output": 10, "cached": 0.13}}
        usage = response.usage
        input_token = int(usage.input_tokens)
        output_token = int(usage.output_tokens)
        cached_tokens = int(usage.input_tokens_details.cached_tokens)
        prices = price_table.get(self.model, None)
        if prices:
            total_price = (
                (input_token - cached_tokens) * prices["input"] / 10**6
                + output_token * prices["output"] / 10**6
                - cached_tokens * prices["cached"] / 10**6
            )
            logger.debug(
                f"Price for this request: ${total_price:.6f} (Input tokens: {input_token}, Output tokens: {output_token}, Cached tokens: {cached_tokens})"
            )
        else:
            logger.warning("Price information not available for this model.")
            total_price = 0
        return total_price

    def prompting(self, user_input, prompt_file, prompt_type, parameters, message_type):
        messages = self.build_openai_messages(
            user_input, prompt_file, prompt_type, parameters, message_type
        )
        one_request_price = 0.0
        if "gpt-5" in self.model and 'gpt-5-mini' not in self.model:
            try:
                response = self.client.responses.create(
                    model="gpt-5-2025-08-07",
                    input=messages,
                    reasoning={"effort": "high"},
                )
                logger.debug(f"The token usage: {response.usage}")
                one_request_price = self.price_calculation(response)
            except openai.BadRequestError as e:
                logger.warning(e)
            logger.debug(f"Response: {response}")
            return_result = response.output_text
            return_result = return_result.strip("```").strip("json")
            logger.debug(return_result)
        # exit()
        elif "gpt-5-mini" in self.model:
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=messages,
                    reasoning={"effort": "medium"},
                )
                logger.debug(f"The token usage: {response.usage}")
                one_request_price = self.price_calculation(response)
            except openai.BadRequestError as e:
                logger.warning(e)
            return_result = response.output_text
            return_result = return_result.strip("```").strip("json")
            logger.debug(return_result)
        else:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    # max_tokens=min(
                    #     len(user_input.split(" ")) * self.max_token_ratio, 16384
                    # ),
                    # response_format=(
                    #     {"type": "json_object"}
                    #     if "json" in str(messages).lower()
                    #     else None
                    # ),
                )
            except openai.BadRequestError as e:
                logger.warning(e)

            logger.debug(f"Response: {response}")
            return_result = response.choices[0].message.content
            return_result = return_result.split("```")[1] if "```" in return_result else return_result
            return_result = return_result.strip("```").strip("json")
            logger.debug(return_result)
        # exit()
        return {"result": return_result, "price": one_request_price}


class QwenPrompter(Prompter):
    def __init__(self, model, config):
        super().__init__(model)
        self.model_obj = AutoModelForCausalLM.from_pretrained(
            config["model_path"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
        self.config = config
        self.max_token_ratio = 5

    def prompting(self, user_input, prompt_file, prompt_type, parameters, message_type):
        messages = self.build_openai_messages(
            user_input, prompt_file, prompt_type, parameters, message_type
        )
        model_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_input = self.tokenizer([model_input], return_tensors="pt").to(
            self.model_obj.device
        )
        generate_ids = self.model_obj.generate(
            **model_input,
            max_new_tokens=min(
                16384, len(user_input.split(" ")) * self.max_token_ratio
            ),
        )
        generate_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_input.input_ids, generate_ids)
        ]
        result = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
        logger.debug(f"Here is the output of {self.model}\n{result}")

        return result


class ClaudePrompter(Prompter):
    def __init__(self, model, config):
        super().__init__(model)
        self.client = anthropic.Anthropic()

    def price_calculation(self, response):
        price_table = {
            "claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
            "claude-opus-4-5": {"input": 5.0, "output": 25.0},
        }
        usage = response.usage
        input_token = int(usage.input_tokens)
        output_token = int(usage.output_tokens)
        prices = price_table.get(self.model, None)
        if prices:
            total_price = (
                input_token * prices["input"] / 10**6
                + output_token * prices["output"] / 10**6
            )
            logger.debug(
                f"Price for this request: ${total_price:.6f} (Input tokens: {input_token}, Output tokens: {output_token})"
            )
        else:
            logger.warning("Price information not available for this model.")
        return total_price

    def prompting(self, user_input, prompt_file, prompt_type, parameters, message_type):
        messages = self.build_openai_messages(
            user_input, prompt_file, prompt_type, parameters, message_type
        )
        system_prompt = [msg["content"] for msg in messages if msg["role"] == "system"]
        messages = [msg for msg in messages if msg["role"] == "user"]
        response = self.client.messages.create(
            model=self.model,
            system=system_prompt[0],
            messages=messages,
            max_tokens=4096,
        )
        logger.debug(f"Response: {response}")
        total_price = self.price_calculation(response)
        return_result = response.content[0].text
        logger.debug(return_result)
        return {"result": return_result, "price": total_price}


class SilliconFlowPrompter(Prompter):
    def __init__(self, model, config):
        super().__init__(model)
        self.model = model
        self.headers = {
            "Authorization": "Bearer sk-ajacybncaebojwfqchfhoyarqamcocputblphevegohofliv",
            "Content-Type": "application/json",
        }

    def prompting(
        self, user_input, sys_prompt, prompt_type, parameters, message_type, url=None
    ):
        messages = self.build_openai_messages(
            user_input, sys_prompt, prompt_type, parameters, message_type
        )
        payload = {
            "model": self.model,
            "messages": messages,
            # "enable_thinking": True,
        }
        if url is not None:
            response = requests.post(url, json=payload, headers=self.headers)
            logger.debug(response)
            logger.debug(response.json())

        max_tries = 100
        attempt = 0
        while attempt < max_tries:
            try:
                result = response.json()["choices"][0]["message"]["content"]
            except json.decoder.JSONDecodeError as e:
                attempt += 1
                logger.debug(e)
                logger.debug(response.text)
                time.sleep(4)
                response = requests.post(url, json=payload, headers=self.headers)
                continue
            if result is None or len(result) == 0:
                logger.exception("Result is None")
                attempt += 1
                time.sleep(4)
                response = requests.post(url, json=payload, headers=self.headers)
                continue
            break
        logger.debug(result)
        # exit()
        return {"result": result.strip("```").strip("json"), "price": 0.0}
