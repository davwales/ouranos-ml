import torch
import gc
from threading import Thread
from transformers import pipeline, TextIteratorStreamer, AutoTokenizer
from accelerate.utils import release_memory
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.prompts.prompt import PromptTemplate

from domain.generation import Message, MessageRole

class GenerateTextQuery:
    def __init__(self, model_name: str, context: str, latest_message: Message, prev_messages: list[Message], human_name: str = "Human", ai_name: str = "AI"):
        self.model_name = model_name
        self.context = context
        self.latest_message = latest_message
        self.prev_messages = prev_messages
        self.human_name = human_name
        self.ai_name = ai_name

    def __enter__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True) # type: ignore
        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            tokenizer=self.tokenizer,
            trust_remote_code=False,
            min_new_tokens=16,
            max_new_tokens=128,
            do_sample=True,
            device_map="auto",
            streamer=self.streamer
        )
        self.hf_pipe = HuggingFacePipeline(pipeline=self.pipe)
        return self

    def __exit__(self, *args):
        release_memory(self.tokenizer)
        release_memory(self.streamer)
        release_memory(self.pipe)
        release_memory(self.hf_pipe)
        self.tokenizer = None
        self.streamer = None
        self.pipe = None
        self.hf_pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def generate(self):
        if self.hf_pipe is None or self.streamer is None:
            print("LangChain pipeline has not been initialized...")
            return

        lc_messages = [self._to_langchain_message(x) for x in self.prev_messages]
        history = ChatMessageHistory(messages=lc_messages)
        memory = ConversationBufferMemory(chat_memory=history, human_prefix=self.human_name, ai_prefix=self.ai_name)

        prompt = self._get_prompt()
        conversation = ConversationChain(llm=self.hf_pipe, prompt=prompt, memory=memory)

        thread = Thread(
            target=conversation.predict,
            kwargs={"input": self.latest_message.content}
        )

        thread.start()
        for text in self.streamer:
            if text is None or text == "":
                continue
            yield text
        thread.join()
    
    def _to_langchain_message(self, m: Message):
        if m.role is MessageRole.USER:
            return HumanMessage(content=m.content, name="test")
        elif m.role is MessageRole.ASSISTANT:
            return AIMessage(content=m.content)
        return SystemMessage(content=m.content)
    
    def _get_prompt(self):
        latest_name = self.human_name if self.latest_message.role is MessageRole.USER else self.ai_name
        template = f"""{self.context}

        Current conversation:
        {{history}}
        {latest_name}: {{input}}
        {self.ai_name}:"""
        return PromptTemplate(input_variables=["history", "input"], template=template)
