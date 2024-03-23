from infrastructure.lc.queries.generate_text_query import GenerateTextQuery

from domain.generation import MessageRole, Message

from .generate_text_command import GenerateTextCommand

class GenerateTextCommandHandler:
    def __init__(self, command: GenerateTextCommand):
        self.command = command

    def generate_text(self):
        if len(self.command.messages) == 0:
            return
        
        system_messages: list[str] = [x.content for x in self.command.messages if x.role is MessageRole.SYSTEM]
        chat_messages: list[Message] = [x for x in self.command.messages if x.role is not MessageRole.SYSTEM]
        
        context: str = " ".join(system_messages)
        latest_message: Message = chat_messages[-1]
        history: list[Message] = chat_messages[:-1]

        with GenerateTextQuery(self.command.model_name, context, latest_message, history, self.command.human_name, self.command.ai_name) as query:
            for chunk in query.generate():
                yield chunk
