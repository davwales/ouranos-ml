from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from accelerate.utils import release_memory
from ..base_generator import BaseGenerator
from ..support.prompt_support import PromptSupport
import torch

class AutoTextGenerator(BaseGenerator):
    model_name: str
    model_revision: str
    max_tokens: int
    
    model: any
    tokenizer: any
    generation_config: GenerationConfig
    prompt_support: PromptSupport

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
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            trust_remote_code=False,
            revision=self.model_revision)
        model = model.to_bettertransformer()
        self.model = model
        
    def unload_model(self):
        release_memory(self.tokenizer)
        release_memory(self.model)
        self.model = None
        self.tokenizer = None
        self.prompt_support = None

    def generate(self, context: list[str], instruction: list[str], response_start: str, extra_stop_words: list[str]):
        if self.model is None or self.tokenizer is None or self.prompt_support is None:
            raise Exception("Attempting to generate from an unloaded model.")

        prompt = self.prompt_support.create_prompt(context, instruction, response_start)
        self.generation_config.stopping_criteria = self.prompt_support.get_stopping_criteria(extra_stop_words)

        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.cuda()
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            output = self.model.generate(inputs=input_ids, generation_config=self.eneration_config)

        raw_output = self.tokenizer.decode(output[0])
        return (input_ids.shape[1], self.prompt_support.clean_response(prompt, raw_output))
