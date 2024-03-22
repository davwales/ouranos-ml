from threading import Thread
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.prompts.prompt import PromptTemplate

from domain.generation import Message, MessageRole
from infrastructure.lc.queries.load_model_query import LoadModelQuery

from .generate_text_command import GenerateTextCommand

class GenerateTextCommandHandler:
    def __init__(self, command: GenerateTextCommand):
        self.command = command

    def generate_text(self):
        if len(self.command.messages) == 0:
            return
        
        latest_message = self.command.messages[-1]
        prev_messages = [self._to_langchain_message(m) for m in self.command.messages[:-1]]

        history = ChatMessageHistory(messages=prev_messages)
        memory = ConversationBufferMemory(chat_memory=history, human_prefix=self.command.human_name, ai_prefix=self.command.ai_name)

        model_query = LoadModelQuery(self.command.model_name)
        prompt = self._get_prompt(latest_message)
        model, streamer = model_query.load_model()
        conversation = ConversationChain(llm=model, prompt=prompt, verbose=True, memory=memory)

        thread = Thread(
            target=conversation.predict,
            kwargs={"input": latest_message.content}
        )

        thread.start()
        for text in streamer:
            cleaned_text = self._clean_text(text)
            if cleaned_text is None or cleaned_text == "":
                continue
            yield text
        thread.join()
    
    def _to_langchain_message(self, m: Message):
        if m.role is MessageRole.USER:
            return HumanMessage(content=m.content, name="test")
        elif m.role is MessageRole.ASSISTANT:
            return AIMessage(content=m.content)
        return SystemMessage(content=m.content)
    
    def _get_prompt(self, latest_message: Message):
        latest_name = self.command.human_name if latest_message.role is MessageRole.USER else self.command.ai_name
        template = f"""The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

        Current conversation:
        {{history}}
        {latest_name}: {{input}}
        {self.command.ai_name}:"""
        return PromptTemplate(input_variables=["history", "input"], template=template)

    def _clean_text(self, text: str):
        cleaned_text = text

        stop_words = ["</s>"]
        for word in stop_words:
            cleaned_text = cleaned_text.replace(word, "")
        return cleaned_text
