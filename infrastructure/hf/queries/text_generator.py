from typing import Any, Dict, Generator
from threading import Thread
from tokenizers import AddedToken
from transformers import pipeline, TextIteratorStreamer, AutoTokenizer, PreTrainedTokenizerBase
from accelerate.utils import release_memory


class TextGenerator:
    def __init__(self, model_id: str):
        self.model_id = model_id
        
    def __enter__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            max_length=2000,
            truncation=True)
        
        self.streamer = TextIteratorStreamer(
            self.tokenizer, # type: ignore
            skip_prompt=True,
            skip_special_tokens=True)
            
        self.pipe = pipeline(
            "text-generation",
            self.model_id,
            tokenizer=self.tokenizer,
            streamer=self.streamer,
            device_map="cuda:0",
            trust_remote_code=False)
        
        return self

    def __exit__(self, *args):
        self.tokenizer, self.streamer, self.pipe = release_memory(
            self.tokenizer, 
            self.streamer, 
            self.pipe)

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        if not isinstance(self.tokenizer, PreTrainedTokenizerBase):
            raise Exception("Unsupported tokenizer...")
        return self.tokenizer

    def generate(self, prompt: str) -> Generator[Any, Any, None]:
        thread = Thread(
            target=self.pipe,
            kwargs={
                "text_inputs": prompt,
                "max_new_tokens": 64
            })

        thread.start()
        for chunk in self.streamer:
            if chunk is None:
                continue
            yield chunk
        thread.join()