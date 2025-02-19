from multiprocessing import Process, Queue
from src.infrastructure.hf.get_prompt import get_prompt
from src.infrastructure.hf.text_generator import TextGenerator

def generate_text(model_id: str, chat_template: str | None, messages: list[dict[str, str]]):
    if len(messages) == 0:
        return
    
    result_queue = Queue()
    p = Process(target=_generate, args=(result_queue, model_id, chat_template, messages))

    p.start()
    while True:
        chunk = result_queue.get() 
        if chunk is None:
            break
        yield chunk
    p.join()

def _generate(result_queue: Queue, model_id: str, chat_template: str | None, messages: list[dict[str, str]]):
    with TextGenerator(model_id) as text_generator:
        prompt = get_prompt(
            text_generator.get_tokenizer(), 
            chat_template, 
            messages
        )

        for chunk in text_generator.generate(prompt):
            if chunk is None:
                continue
            result_queue.put(chunk)
        result_queue.put(None)
