class GenerateTextQuery:
    def __init__(self, model_id: str, chat_template: str | None, messages: list[dict[str, str]]):
        self.model_id = model_id
        self.chat_template = chat_template
        self.messages = messages
