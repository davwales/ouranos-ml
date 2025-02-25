from typing import Generator, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase, TextIteratorStreamer
from threading import Thread
import torch
import gc

class TextGenerator:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        
    def __enter__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            max_length=2000,
            truncation=True,
            use_fast=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        return self

    def __exit__(self, *args):
        if hasattr(self.model, "to"):
            self.model.to("cpu")
        
        del self.model
        del self.tokenizer
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        if not isinstance(self.tokenizer, PreTrainedTokenizerBase):
            raise Exception("Unsupported tokenizer...")
        return self.tokenizer

    def generate(self, prompt: str) -> Generator[Any, Any, None]:
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen_kwargs = dict(
            input_ids=input_ids.input_ids,
            min_new_tokens=16,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.1,
            do_sample=True,
            streamer=streamer
        )

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        for token in streamer:
            if token is not None:
                yield token