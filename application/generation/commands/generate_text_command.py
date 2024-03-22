from domain.generation import Message

class GenerateTextCommand:
    def __init__(self, model_name: str, messages: list[Message], human_name: str = "Human", ai_name: str = "AI"):
        self.model_name = model_name
        self.messages = messages
        self.human_name = human_name
        self.ai_name = ai_name
