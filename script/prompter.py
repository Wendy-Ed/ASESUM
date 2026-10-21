import os
import openai
import torch

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
        prompt_template = load_prompt_template(os.path.join(PROMPT_DIR, prompt_file), prompt_type)
        if parameter_dict is not None:
            prompt = prompt_template.format(**parameter_dict)
        else:
            prompt = prompt_template

        logger.debug(f"Here is the prompt\n{prompt}")
        return prompt

    def build_openai_messages(
        self, user_input, prompt_file, prompt_type, parameters, message_type
    ):
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

    def prompting(self, user_input, prompt_file, prompt_type, parameters, message_type):
        messages = self.build_openai_messages(
            user_input, prompt_file, prompt_type, parameters, message_type
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=min(len(user_input.split(" ")) * self.max_token_ratio, 16384),
                response_format=(
                    {"type": "json_object"} if "json" in str(messages).lower() else None
                ),
            )
        except openai.BadRequestError as e:
            logger.warning(e)

        return_result = response.choices[0].message.content
        return_result = return_result.strip("```").strip('json')
        logger.debug(return_result)
        return return_result
    
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
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_input = self.tokenizer([model_input], return_tensors="pt").to(self.model_obj.device)
        generate_ids = self.model_obj.generate(**model_input, max_new_tokens=min(16384, len(user_input.split(" ")) * self.max_token_ratio))
        generate_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_input.input_ids, generate_ids)
        ]
        result = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
        logger.debug(f"Here is the output of {self.model}\n{result}")

        return result