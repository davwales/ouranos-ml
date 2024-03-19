from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from accelerate.utils import release_memory
from application.generation import BaseGenerator, PromptSupport
import torch

class PipelineTextGenerator(BaseGenerator):
    model_name: str
    model_revision: str
    max_tokens: int

    model: any
    pipe: any
    prompt_support: PromptSupport
    generation_config: GenerationConfig
    tokenizer: any

    def __init__(self, model_name: str, model_revision: str = "main", max_tokens: int = 1250):
        self.model_name = model_name
        self.model_revision = model_revision
        self.max_tokens = max_tokens
        self.generation_config = GenerationConfig(
            min_new_tokens=16,
            max_new_tokens=128,
            do_sample=True,
            use_cache=True,
            temperature=1.31,
            top_p=0.29,
            top_k=72,
            repetition_penalty=1.09
        )
        
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.prompt_support = PromptSupport(self.tokenizer, self.max_tokens)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            trust_remote_code=False,
            revision=self.model_revision)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto")
        
    def unload_model(self):
        release_memory(self.tokenizer)
        release_memory(self.model)
        release_memory(self.pipe)
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.prompt_support = None

    def generate(self, context: list[str], instruction: list[str], response_start: str, extra_stop_words: list[str]):
        if self.model is None or self.tokenizer is None or self.prompt_support is None:
            raise Exception("Attempting to generate from an unloaded model.")

        prompt = self.prompt_support.create_prompt(context, instruction, response_start)
        self.generation_config.stopping_criteria = self.prompt_support.get_stopping_criteria(extra_stop_words)

        tokens = self.tokenizer(prompt, return_tensors='pt').input_ids.shape[1]
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            output = self.pipe(prompt, generation_config=self.generation_config)

        raw_output = output[0]["generated_text"]
        return (tokens, self.prompt_support.clean_response(prompt, raw_output))
