from transformers import AutoTokenizer, StoppingCriteriaList

class PromptSupport:
    tokenizer: AutoTokenizer
    max_tokens: int
    stop_tokens: list

    def __init__(self, tokenizer: AutoTokenizer, max_tokens: int):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.stop_tokens = self._get_stop_words(["</s>", "\n"])

    def create_prompt(self, context: list[str], instruction: list[str], response_start: str):
        full_context = "\n".join(context)
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

        ### Instruction:
        {full_context}

        ### Input:
        [INSTRUCTIONS]

        ### Response:
        {response_start}
        """
        
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        remaining_tokens = self.max_tokens - prompt_tokens
        
        included_instructions: list[str] = []
        for i in reversed(instruction):
            instruction_tokens = self.tokenizer(i, return_tensors="pt").input_ids.shape[1]
            if instruction_tokens > remaining_tokens:
                break
            included_instructions.append(i)
            remaining_tokens -= instruction_tokens
        prompt = prompt.replace("[INSTRUCTIONS]", "\n".join(reversed(included_instructions)))
        return prompt


    def clean_response(self, prompt: str, response: str):
        response = response.replace(prompt, "")
        for w in ["<s>","</s>","\n"]:
            response = response.replace(w, "")
        return response.strip()
    
    def get_stopping_criteria(self, words: list[str] = None):
        all_tokens = self.stop_tokens.copy()
        if words and len(words) > 0:
            new_tokens = self._get_stop_words(words)
            all_tokens.extend(new_tokens)
        return StoppingCriteriaList(all_tokens)
    
    def _get_stop_words(self, words: list[str]):
        return [self.tokenizer(w, return_tensors='pt').input_ids.cuda() for w in words]
