from src.domain.chat.chat_message import ChatMessage
from src.infrastructure.hf.get_prompt import get_prompt
from src.infrastructure.hf.text_generator import TextGenerator

def generate_text(model_id: str, chat_template: str | None, messages: list[ChatMessage]):
    if len(messages) == 0:
        return
    
    with TextGenerator(model_id) as text_generator:
        prompt = get_prompt(
            text_generator.get_tokenizer(), 
            chat_template, 
            messages
        )

        for chunk in text_generator.generate(prompt):
            if chunk is None:
                continue
            yield chunk
