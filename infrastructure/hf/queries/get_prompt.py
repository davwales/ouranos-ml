from transformers import PreTrainedTokenizerBase

def get_prompt(tokenizer: PreTrainedTokenizerBase, chat_template: str | None, messages: list[dict[str, str]]) -> str:
    cleaned_messages = _combine_consecutive_messages(messages)
    
    if chat_template is not None:
         tokenizer.chat_template = chat_template

    prompt = tokenizer.apply_chat_template(cleaned_messages, tokenize=False)
    if not isinstance(prompt, str):
         raise Exception("Generated prompt is not a string.")
    
    return prompt

def _combine_consecutive_messages(messages: list[dict[str, str]]):
    if not messages:
        return []

    combined_messages = [messages[0]]
    for message in messages[1:]:
        role, content = message['role'], message['content']
        prev_role, prev_content = combined_messages[-1]['role'], combined_messages[-1]['content']
        if role == prev_role:
            combined_messages[-1]['content'] += f" {content}"
        else:
            combined_messages.append({'role': role, 'content': content})
    return combined_messages
