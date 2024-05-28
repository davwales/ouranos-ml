from multiprocessing import Process, Queue
from infrastructure.hf.queries.get_prompt import get_prompt
from infrastructure.hf.queries.text_generator import TextGenerator

from .generate_text_query import GenerateTextQuery

class GenerateTextQueryHandler:
    def __init__(self, query: GenerateTextQuery):
        self.query = query

    def generate_text(self):
        if len(self.query.messages) == 0:
            return
        
        result_queue = Queue()
        p = Process(target=self._generate, args=(result_queue,))

        p.start()
        while True:
            chunk = result_queue.get() 
            if chunk is None:
                break
            yield chunk
        p.join()

    def _generate(self, result_queue: Queue):
        with TextGenerator(self.query.model_id) as text_generator:
            prompt = get_prompt(
                text_generator.get_tokenizer(), 
                self.query.chat_template, 
                self.query.messages)

            for chunk in text_generator.generate(prompt):
                if chunk is None:
                    continue
                result_queue.put(chunk)
            result_queue.put(None)
