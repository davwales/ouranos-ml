from transformers import pipeline, TextIteratorStreamer, AutoTokenizer
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

class LoadModelQuery:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def load_model(self):
        print("Loading LangChain model...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
        pipe = pipeline(
            "text-generation",
            model=self.model_name,
            tokenizer=tokenizer,
            trust_remote_code=False,
            min_new_tokens=16,
            max_new_tokens=128,
            do_sample=True,
            device_map="auto",
            streamer=streamer
        )
        hf_pipe = HuggingFacePipeline(pipeline=pipe)
        return hf_pipe, streamer
