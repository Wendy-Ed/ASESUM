from loguru import logger
from utils.preprocessing import build_review_context, string_to_list, string_to_json, process_result


class BaseExtractor:
    def __init__(self, prompt, prompt_type, prompter=None):
        self.prompt_path = prompt
        self.prompt_type = prompt_type
        self.prompter = prompter

    def extract(self, data, model_name, **kwargs):
        raise NotImplementedError


class BaselineExtrator(BaseExtractor):
    def __init__(self, prompt, prompt_type, prompter=None):
        super().__init__(prompt, prompt_type, prompter)

    def extract(self, data, model_name, **kwargs):
        """Input the data df and generate the summaries with all reviews inside"""
        logger.debug(model_name)
        logger.debug(kwargs)
        data_string = build_review_context(
            data[kwargs["column"]].values.tolist(),
            kwargs["context_length"] - data.shape[0],
        )
        logger.debug(f"The length of all reviews: {len(data_string.split(' '))}")

        generated_summary = self.prompter.prompting(
            data_string, self.prompt_path, self.prompt_type, {}, "system"
        )

        logger.debug(generated_summary)
        return generated_summary


class AspectExtractor(BaseExtractor):
    def __init__(self, prompt, prompt_type, prompter):
        super().__init__(prompt, prompt_type, prompter)

    def aspect_extraction(self, review, aspect_set):
        # sort the aspect dict by the mention of aspect
        aspect_set_order = dict(
            sorted(aspect_set.items(), key=lambda i: i[1], reverse=True)
        )
        # take the top 15 most common aspects in the prompt
        top_aspects = ", ".join(list(aspect_set_order.keys())[:15])
        logger.debug(f"Here are the aspects {top_aspects}")
        raw_string = self.prompter.prompting(
            review,
            self.prompt_path,
            self.prompt_type,
            {"aspect": top_aspects},
            "system",
        )
        result_list = string_to_json(raw_string)
        logger.debug(result_list)
        if aspect_set is not None:
            process_result(result_list, aspect_set)
        return result_list

    def extract(self, data, **kwargs):
        if "seed_aspects" in kwargs:
            full_aspect = {k: 0 for k in kwargs["seed_aspects"]}
        elif "category" in kwargs:
            full_aspect = {k: 0 for k in kwargs["category"]}
        assert "column" in kwargs
        column = kwargs["column"]

        if kwargs["column"] == "text":
            assert column in data.columns
            data["aspects"] = data[column].apply(
                lambda x: (
                    self.aspect_extraction(
                        x,
                        aspect_set=full_aspect,
                    )
                    if x is not None
                    else x
                )
            )

        else:
            data_string = ", ".join(
                data["product_meta"]["categories"][
                    : int(len(data["product_meta"]["categories"]) / 2)
                ]
            )
            main_aspects = self.prompter.prompting(
                data_string, "aspect_initiate.txt", self.prompt_type, {}, "system"
            )

            return string_to_list(main_aspects)
