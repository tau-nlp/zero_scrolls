import os
import sys

import anthropic

sys.path.append(os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from experiments.api.api import APIRunner

CLAUDE_V1 = "claude_v1"

class AnthropicAPI(APIRunner):

    def __init__(self, model_name: str, dataset_name: str, format_prompt_loc: str = "start",
                 max_input_output_tokens: int = None):
        super().__init__(model_name, dataset_name)
        self._client = None
        self.top_p = -1
        self.temperature = 1  # https://console.anthropic.com/docs/api/reference
        self.top_k = 1

        self.max_tokens = 8000
        self.tags_in_prompt = True


    @property
    def max_input_output_tokens(self):
        return self.max_tokens  # https://console.anthropic.com/docs/prompt-design#prompt-length

    def init_api(self):
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.tokenizer = self._client.get_tokenizer()

    @property
    def max_generation_tokens_key(self):
        return "max_tokens_to_sample"

    @property
    def max_generation_tokens(self):
        return super().max_generation_tokens  # + 64

    def get_number_of_tokens(self, string):
        return self._client.count_tokens(string)

    def parse_finish_reason(self, response):
        return response.stop_reason

    def init_params(self):
        params = super().init_params()
        params["stop_sequences"] = [anthropic.HUMAN_PROMPT]
        return params

    def build_prompt(self, example):

        if self.dataset_name in APIRunner.summarization_datasets:
            anthropic_suffix = anthropic.AI_PROMPT
        else:
            format_request, tag, answer_type = self.get_chat_format_request_tag_and_answer_type(example)
            self.insert_format_request(example, format_request)
            anthropic_suffix = f"{anthropic.AI_PROMPT}"
            if self.tags_in_prompt:
                anthropic_suffix += f" {answer_type}: <{tag}>"

        input_without_suffix = example['input'][:example['query_end_index']]
        anthropic_prompt = f"{anthropic.HUMAN_PROMPT} {input_without_suffix}{anthropic_suffix}"  # https://console.anthropic.com/docs/prompt-design/classification

        tokenized = self.tokenizer.encode(anthropic_prompt)
        if len(tokenized.ids) <= self.max_input_output_tokens - self.max_generation_tokens:
            return anthropic_prompt

        query = example['input'][example['query_start_index']:example['query_end_index']]
        truncation_seperator = example['truncation_seperator']
        anthropic_suffix = f"{truncation_seperator}{query}{anthropic_suffix}"
        anthropic_prefix_and_suffix_tokenized = self.tokenizer.encode(
            f"{anthropic.HUMAN_PROMPT} {anthropic_suffix}").ids
        max_tokens_for_input = self.get_max_document_tokens(len(anthropic_prefix_and_suffix_tokenized))
        char_idx_of_max_tokens_for_input = tokenized.offsets[max_tokens_for_input][0]
        input_without_suffix_trimmed = input_without_suffix[:char_idx_of_max_tokens_for_input]
        anthropic_prompt = f"{anthropic.HUMAN_PROMPT} {input_without_suffix_trimmed}{anthropic_suffix}"
        return anthropic_prompt

    def parse_prediction(self, response):
        return response.completion

    def parse_model_name(self, parameters, response):
        return response.model

    def call(self, parameters):
        return self._client.completions.create(**parameters)

    def get_chat_format_request_tag_and_answer_type(self, example):
        format_request, tag, answer_type = super().get_chat_format_request_tag_and_answer_type(example)
        if self.tags_in_prompt:
            if format_request[-1] == ".":
                format_request = format_request[:-1]
            format_request += f", and please highlight your final {answer_type.lower()} with <{tag}></{tag}> tags."
        return format_request, tag, answer_type
