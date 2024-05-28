from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from application.generation.queries import GenerateTextQuery, GenerateTextQueryHandler

from .requests import TextGenerationRequest

router = APIRouter(prefix="/generation")

chatml_template = """
    {% for message in messages %}
        {{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
    {% endfor %}
"""

alpaca_template = """
{{ (messages|selectattr('role', 'equalto', 'system')|list|last).content|trim if (messages|selectattr('role', 'equalto', 'system')|list) else '' }}

{% for message in messages %}
{% if message['role'] == 'user' %}
### Instruction:
{{ message['content']|trim -}}
{% if not loop.last %}

{% endif %}
{% elif message['role'] == 'assistant' %}
### Response:
{{ message['content']|trim -}}
{% if not loop.last %}

{% endif %}
{% endif %}
{% endfor %}
{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}
### Response:
{% endif %}
"""

@router.post("/text")
def text(request: TextGenerationRequest):
    print("Generating text...")

    model = "TheBloke/Kunoichi-7B-GPTQ"

    query = GenerateTextQuery(model, alpaca_template, request.messages)
    command_handler = GenerateTextQueryHandler(query)

    return StreamingResponse(command_handler.generate_text(), media_type="text/plain")
