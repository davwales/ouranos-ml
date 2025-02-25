from transformers import PreTrainedTokenizerBase

from src.domain.chat.chat_message import ChatMessage

def get_prompt(tokenizer: PreTrainedTokenizerBase, chat_template: str | None, messages: list[ChatMessage]) -> str:
    cleaned_messages = _combine_consecutive_messages(messages)
    
    if chat_template is not None:
         tokenizer.chat_template = chat_template

    input = [{"role": message.Role, "content": message.Content} for message in cleaned_messages]
    prompt = tokenizer.apply_chat_template(input, tokenize=False)
    if not isinstance(prompt, str):
         raise Exception("Generated prompt is not a string.")
    
    return prompt

def _combine_consecutive_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    if not messages:
        return []

    combined_messages = [messages[0]]
    for message in messages[1:]:
        role, content = message.Role, message.Content
        prev_role = combined_messages[-1].Role
        if role == prev_role:
            combined_messages[-1].Content += f" {content}"
        else:
            combined_messages.append(ChatMessage(Role=role, Content=content))
    return combined_messages
