from pydantic import AliasGenerator, BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class BaseSchema(BaseModel):
    """Base schema for all models. Used to apply global configuration.

    Input accepts both camelCase (the alias) and snake_case (the field name,
    via populate_by_name). Serialization always uses the snake_case field name
    so that every response path (FastAPI's by_alias JSON encoding and the SSE
    model_dump_json path) emits the OpenAI wire-style casing pantheon expects.
    """

    model_config = ConfigDict(
        alias_generator=AliasGenerator(alias=to_camel, serialization_alias=lambda field_name: field_name),
        populate_by_name=True,
        from_attributes=True,
    )
