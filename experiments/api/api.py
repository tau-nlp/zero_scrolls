from datetime import datetime

dataset_to_leave_tokens_for_generations = {
    "gov_report": 1024,
    "summ_screen_fd": 512,
    "qmsum": 512,
    "qasper": 128,
    "narrative_qa": 64,
    "quality": 10,
    "musique": 32,
    "squality": 512,
    "space_digest": 36,
    "book_sum_sort": 256,
}

class APIRunner:
    summarization_datasets = ["gov_report", "summ_screen_fd", "qmsum", "squality"]

    def __init__(self, model_name: str, dataset_name: str, min_ms_between_api_calls: int = 20):
        self.top_k = None
        self.temperature = None  # meaning could change from one api to another
        self.top_p = None
        self.model_name = model_name
        self.min_ms_between_api_calls = min_ms_between_api_calls
        self.dataset_name = dataset_name
        self.init_api()

    @property
    def max_input_output_tokens(self):
        # max tokens for input + output
        raise NotImplementedError("max_input_output_tokens")

    def init_api(self):
        raise NotImplementedError("init_api")

    def init_params(self):
        params = {
            self.max_generation_tokens_key: self.max_generation_tokens,
            "model": self.model_name
        }

        if self.temperature is not None:
            params["temperature"] = self.temperature

        if self.top_k is not None:
            params[self.top_k_key] = self.top_k

        if self.top_p is not None:
            params[self.top_p_key] = self.top_p

        return params

    @property
    def max_generation_tokens_key(self):
        raise NotImplementedError("max_generation_tokens_key_name")

    @property
    def top_k_key(self):
        return "top_k"

    @property
    def top_p_key(self):
        return "top_p"

    @property
    def max_generation_tokens(self):
        return dataset_to_leave_tokens_for_generations[self.dataset_name]

    def get_number_of_tokens(self, string):
        raise NotImplementedError("get_number_of_tokens")

    def get_max_document_tokens(self, n_suffix_tokens):
        return self.max_input_output_tokens - n_suffix_tokens - self.max_generation_tokens - 1

    def parse_finish_reason(self, response):
        raise NotImplementedError("parse_finish_reason")

    def parse_prediction(self, response):
        raise NotImplementedError("parse_prediction")

    def build_prompt(self, example, ):
        raise NotImplementedError("build_input")

    def preprocess_parameters(self, parameters, prompt):
        parameters["prompt"] = prompt

    def parse_model_name(self, parameters, response):
        raise NotImplementedError("build_input")

    def build_output(self, example, prompt, parameters, response):
        prediction = self.parse_prediction(response)
        output = {
            "id": example["id"],
            "model": self.parse_model_name(parameters, response),
            "original_example_input": example["input"],
            "prompt": prompt,
            "max_input_output_tokens": self.max_input_output_tokens,
            "n_input_tokens": self.get_number_of_tokens(prompt),
            "n_generated_tokens": self.get_number_of_tokens(prediction),
            "finish_reason": self.parse_finish_reason(response),
            "temperature": parameters["temperature"]
        }

        if self.top_k_key in parameters:
            output[self.top_k_key] = parameters[self.top_k_key]

        if self.top_p_key in parameters:
            output[self.top_p_key] = parameters[self.top_p_key]

        output["prediction"] = prediction

        return output

    def call(self, prompt):
        raise NotImplementedError("call")

    def get_chat_format_request_tag_and_answer_type(self, example):
        answer_type = example["input"][example['query_end_index']:].strip().replace(":", "")
        tag = answer_type.lower().replace(" ", "_")
        format_request = "Do not provide any explanation."
        return format_request, tag, answer_type

    def insert_format_request(self, example, format_request):
        instruction_end_index = example['input'].find("\n\n")
        instruction = example['input'][:instruction_end_index]
        example['input'] = f"{instruction.strip()} {format_request}{example['input'][instruction_end_index:]}"

        for key in ["document_start_index", "document_end_index", "query_start_index", "query_end_index"]:
            example[key] += len(format_request) + 1


def _ms_since_epoch():
    epoch = datetime.utcfromtimestamp(0)
    now = datetime.utcnow()
    return int((now - epoch).total_seconds() * 1000)
