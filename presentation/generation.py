from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from application.generation.queries import GenerateTextQuery, GenerateTextQueryHandler

from .requests import TextGenerationRequest

router = APIRouter(prefix="/generation")

chatml_template = """
    {% for message in messages %}
        {{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
    {% endfor %}
    <|im_start|>assistant
    
"""

alpaca_template = """
{% if messages[0]['role'] == 'system' %}
    {% set system_message = messages[0]['content'] | trim + '\n\n' %}
    {% set messages = messages[1:] %}
{% else %}
    {% set system_message = '' %}
{% endif %}

{{ bos_token + system_message }}
{% for message in messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}

    {% if message['role'] == 'user' %}
        {{ '### Instruction:\n' + message['content'] | trim + '\n\n' }}
    {% elif message['role'] == 'assistant' %}
        {{ '### Response:\n' + message['content'] | trim + eos_token + '\n\n' }}
    {% endif %}
{% endfor %}

{% if add_generation_prompt %}
    {{ '### Instruction:\n' }}
{% endif %}

### Response: 
"""

llama3_template = """
<|begin_of_text|>
{% for message in messages %}
{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>' + message['content'] + '<|eot_id|>' }}
{% endfor %}
"""

vicuna_template = """
{% for message in messages %}
{{ message['role'] + ': ' + message['content'] }}
{% endfor %}
"""

@router.post("/text")
def text(request: TextGenerationRequest):
    print("Generating text...")

    model = "TheBloke/Loyal-Macaroni-Maid-7B-GPTQ"

    query = GenerateTextQuery(model, alpaca_template, request.messages)
    command_handler = GenerateTextQueryHandler(query)

    return StreamingResponse(command_handler.generate_text(), media_type="text/plain")
