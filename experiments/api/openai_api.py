import os
import sys

import openai
import tiktoken

sys.path.append(os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from experiments.api.api import APIRunner

DAVINCI = "text_davinci_003"
ChatGPT = "gpt_3.5_turbo"
GPT4 = "gpt_4"
DAVINCI_MAX_INPUT_OUTPUT_TOKENS = 4096
GPT4_MAX_INPUT_OUTPUT_TOKENS = 8192


class OpenAIAPI(APIRunner):

    def __init__(self, model_name: str, dataset_name: str):
        super().__init__(model_name, dataset_name)
        self.temperature = 0
        # self.top_p = 0
        self.top_k = None
        self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        model_name_underscore = self.model_name.replace("-", "_")
        self.is_chat_api = model_name_underscore in {ChatGPT, GPT4}
        self.is_gpt4 = model_name_underscore == GPT4

    @property
    def max_input_output_tokens(self):
        return GPT4_MAX_INPUT_OUTPUT_TOKENS if self.is_gpt4 else DAVINCI_MAX_INPUT_OUTPUT_TOKENS

    def init_api(self):
        openai.organization = os.getenv("OPENAI_ORG")
        openai.api_key = os.getenv("OPENAI_API_KEY")

    @property
    def max_generation_tokens_key(self):
        return "max_tokens"

    def parse_finish_reason(self, response):
        return response.choices[0]["finish_reason"]

    def build_output(self, example, prompt, parameters, response):
        output = super().build_output(example, prompt, parameters, response)

        output.update({
            "index": response.choices[0]["index"],
        })

        return output

    def parse_model_name(self, parameters, response):
        return response.model

    def build_prompt(self, example):
        if self.is_chat_api:
            if self.dataset_name not in APIRunner.summarization_datasets:
                format_request, _, answer_type = self.get_chat_format_request_tag_and_answer_type(example)
                self.insert_format_request(example, format_request)

            example["input"] = example["input"][:example['query_end_index']]
        tokenized = self.tokenizer.encode(example['input'])
        if len(tokenized) <= self.max_input_output_tokens - self.max_generation_tokens:
            return example['input']

        query_and_answer_prompt = example['input'][example['query_start_index']:]
        truncation_seperator = example['truncation_seperator']

        suffix_tokenized = self.tokenizer.encode(truncation_seperator + query_and_answer_prompt)

        max_tokens_for_input = self.get_max_document_tokens(len(suffix_tokenized))

        tokenized_trimmed = tokenized[:max_tokens_for_input]
        prompt = self.tokenizer.decode(tokenized_trimmed) + truncation_seperator + query_and_answer_prompt

        return prompt

    def get_max_document_tokens(self, n_suffix_tokens):
        max_input_tokens = super().get_max_document_tokens(n_suffix_tokens)
        if self.is_chat_api:
            max_input_tokens -= 9
        return max_input_tokens
    def preprocess_parameters(self, parameters, prompt):
        if self.is_chat_api:
            parameters["messages"] = [
                {"role": "user", "content": prompt}
            ]

            if self.is_gpt4:
                parameters["model"] = GPT4.replace("_", "-")

        else:
            super().preprocess_parameters(parameters, prompt)

    def get_number_of_tokens(self, prompt):
        return len(self.tokenizer.encode(prompt))

    def parse_prediction(self, response):
        if self.is_chat_api:
            return response.choices[0].message.content
        return response.choices[0].text

    def call(self, parameters):
        if self.is_chat_api:
            return openai.ChatCompletion.create(**parameters)
        return openai.Completion.create(**parameters)
