from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, StoppingCriteriaList, pipeline
from accelerate.utils import release_memory
from ..base_generator import BaseGenerator
import torch

class PipelineTextGenerator(BaseGenerator):
    model_name: str
    model_revision: str
    model: any
    pipe: any
    generation_config: GenerationConfig
    tokenizer: any
    max_tokens: int

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

    def generate(self, context: list[str], instruction: list[str], response_start: str, extra_stop_words: list[str]):
        if self.model is None or self.tokenizer is None:
            raise Exception("Attempting to generate from an unloaded model.")

        prompt = self.__create_prompt(context, instruction, response_start)
        stop_words = ["</s>", "\n"]
        stop_words.extend(extra_stop_words)
        stop_word_ids = [self.tokenizer(stop_word, return_tensors='pt').input_ids.cuda() for stop_word in stop_words]
        self.generation_config.stopping_criteria = StoppingCriteriaList(stop_word_ids)

        tokens = self.tokenizer(prompt, return_tensors='pt').input_ids.shape[1]
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            output = self.pipe(prompt, generation_config=self.generation_config)

        raw_output = output[0]["generated_text"]
        return (tokens, self.__clean_response(prompt, raw_output))

    def __create_prompt(self, context: list[str], instruction: list[str], response_start: str):
        full_context = "\n".join(context)
        prompt = f"""
        {full_context}

        ### Instruction:
        [INSTRUCTIONS]

        ### Response:
        {response_start}
        """
        
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        remaining_tokens = self.max_tokens - prompt_tokens
        
        included_instructions: list[str] = []
        for i in reversed(instruction):
            instruction_tokens = self.tokenizer(i, return_tensors="pt").input_ids.shape[1]
            if instruction_tokens > remaining_tokens:
                break
            included_instructions.append(i)
            remaining_tokens -= instruction_tokens
        prompt = prompt.replace("[INSTRUCTIONS]", "\n".join(reversed(included_instructions)))
        return prompt


    def __clean_response(self, prompt: str, response: str):
        response = response.replace(prompt, "")
        for w in ["<s>","</s>","\n"]:
            response = response.replace(w, "")
        return response.strip()
