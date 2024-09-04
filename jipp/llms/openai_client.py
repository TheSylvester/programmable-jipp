import os
from typing import Union, Optional, Dict, List, Any, Tuple
from pydantic import BaseModel
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
)
from openai.types.chat import ChatCompletionMessage
from openai.types.chat_model import ChatModel
from openai._types import NotGiven, NOT_GIVEN
from openai.lib._tools import pydantic_function_tool

from jipp.llms.pydantic_to_schema import pydantic_model_to_openai_schema
from jipp.models.jipp_models import (
    ToolCall,
    Function,
    CompletionUsage,
    LLMError,
    LLMMessage,
    LLMResponse,
)

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def ask_openai(
    *,
    model: Union[str, ChatModel],
    messages: List[LLMMessage],
    response_format: type[BaseModel] | NotGiven = NOT_GIVEN,
    temperature: Optional[float] | NotGiven = NOT_GIVEN,
    top_p: Optional[float] | NotGiven = NOT_GIVEN,
    n: Optional[int] | NotGiven = NOT_GIVEN,
    max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
    stop: Union[Optional[str], List[str]] | NotGiven = NOT_GIVEN,
    presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
    frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
    logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
    tools: List[BaseModel] | NotGiven = NOT_GIVEN,
    tool_choice: Optional[str] | NotGiven = NOT_GIVEN,
    seed: Optional[int] | NotGiven = NOT_GIVEN,
    user: str | NotGiven = NOT_GIVEN,
    api_key: Optional[str] = None,
    organization: Optional[str] = None,
    timeout: float | NotGiven = NOT_GIVEN,
) -> LLMResponse:
    try:

        # Convert pydantic models to function tools with openai.lib._tools.pydantic_function_tool
        if tools is not NOT_GIVEN:
            processed_tools = [
                (
                    pydantic_function_tool(tool)
                    if isinstance(tool, type) and issubclass(tool, BaseModel)
                    else tool
                )
                for tool in tools
            ]
        else:
            processed_tools = NOT_GIVEN

        kwargs = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "max_tokens": max_tokens,
            "stop": stop,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "logit_bias": logit_bias,
            "tools": processed_tools,
            "tool_choice": tool_choice,
            "seed": seed,
            "user": user,
            "timeout": timeout,
        }

        # Handle response_format separately
        if response_format is not NOT_GIVEN:
            kwargs["response_format"] = pydantic_model_to_openai_schema(response_format)

        # Remove NOT_GIVEN values
        kwargs = {k: v for k, v in kwargs.items() if v is not NOT_GIVEN}

        # Call OpenAI API
        response = await client.chat.completions.create(**kwargs)

        # Convert ChatCompletion response to LLMResponse
        return convert_chat_completion_to_llm_response(response)

    except Exception as e:
        # Convert OpenAI-specific errors to a generic LLMError
        raise LLMError(f"Error in OpenAI API call: {str(e)}")


def convert_to_llm_message(
    message: ChatCompletionMessage,
) -> LLMMessage:

    llm_message = LLMMessage(role="assistant", content=message.content)

    if message.function_call:
        llm_message.function_call = Function(
            name=message.function_call.name, arguments=message.function_call.arguments
        )

    if message.tool_calls:
        llm_message.tool_calls = [
            ToolCall(
                id=tool_call.id,
                type=tool_call.type,
                function=Function(
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                ),
            )
            for tool_call in message.tool_calls
        ]

    return llm_message


def convert_chat_completion_to_llm_response(
    chat_completion: ChatCompletion,
) -> LLMResponse:

    choice = chat_completion.choices[0]
    message = convert_to_llm_message(choice.message)

    return LLMResponse(
        message=message,
        usage=(
            CompletionUsage(
                prompt_tokens=chat_completion.usage.prompt_tokens,
                completion_tokens=chat_completion.usage.completion_tokens,
                total_tokens=chat_completion.usage.total_tokens,
            )
            if chat_completion.usage
            else None
        ),
        model=chat_completion.model,
        finish_reason=choice.finish_reason,
    )


# import pydantic


# def pydantic_model_to_openai_schema(
#     model: Union[type[pydantic.BaseModel], pydantic.TypeAdapter[Any]]
# ) -> Dict[str, Any]:
#     """Convert a Pydantic model to OpenAI's strict JSON schema format."""

#     def ensure_strict_schema(
#         schema: Dict[str, Any], path: Tuple[str, ...] = (), root: Dict[str, Any] = None
#     ) -> Dict[str, Any]:
#         if not isinstance(schema, dict):
#             raise TypeError(f"Expected {schema} to be a dictionary; path={path}")

#         root = root or schema

#         # Handle $defs
#         defs = schema.get("$defs")
#         if isinstance(defs, dict):
#             for def_name, def_schema in defs.items():
#                 ensure_strict_schema(
#                     def_schema, path=(*path, "$defs", def_name), root=root
#                 )

#         # Handle type-specific logic
#         if schema.get("type") == "object":
#             if "additionalProperties" not in schema:
#                 schema["additionalProperties"] = False

#             properties = schema.get("properties", {})
#             if properties:
#                 schema["required"] = list(properties.keys())
#                 schema["properties"] = {
#                     key: ensure_strict_schema(
#                         prop_schema, path=(*path, "properties", key), root=root
#                     )
#                     for key, prop_schema in properties.items()
#                 }

#         elif schema.get("type") == "array" and isinstance(schema.get("items"), dict):
#             schema["items"] = ensure_strict_schema(
#                 schema["items"], path=(*path, "items"), root=root
#             )

#         # Handle unions (anyOf)
#         any_of = schema.get("anyOf")
#         if isinstance(any_of, list):
#             schema["anyOf"] = [
#                 ensure_strict_schema(variant, path=(*path, "anyOf", str(i)), root=root)
#                 for i, variant in enumerate(any_of)
#             ]

#         # Handle intersections (allOf)
#         all_of = schema.get("allOf")
#         if isinstance(all_of, list):
#             if len(all_of) == 1:
#                 schema.update(
#                     ensure_strict_schema(
#                         all_of[0], path=(*path, "allOf", "0"), root=root
#                     )
#                 )
#                 schema.pop("allOf")
#             else:
#                 schema["allOf"] = [
#                     ensure_strict_schema(
#                         entry, path=(*path, "allOf", str(i)), root=root
#                     )
#                     for i, entry in enumerate(all_of)
#                 ]

#         # Remove None defaults
#         if schema.get("default") is None:
#             schema.pop("default", None)

#         # Handle $ref
#         ref = schema.get("$ref")
#         if ref and len(schema) > 1:
#             resolved = resolve_ref(root, ref)
#             schema.update({**resolved, **schema})
#             schema.pop("$ref")

#         return schema

#     def resolve_ref(root: Dict[str, Any], ref: str) -> Dict[str, Any]:
#         if not ref.startswith("#/"):
#             raise ValueError(f"Unexpected $ref format {ref!r}; Does not start with #/")

#         path = ref[2:].split("/")
#         resolved = root
#         for key in path:
#             resolved = resolved[key]
#             if not isinstance(resolved, dict):
#                 raise ValueError(
#                     f"Expected `$ref: {ref}` to resolve to a dictionary but got {resolved}"
#                 )

#         return resolved

#     if isinstance(model, type) and issubclass(model, pydantic.BaseModel):
#         json_schema = model.model_json_schema()
#     elif isinstance(model, pydantic.TypeAdapter):
#         json_schema = model.json_schema()
#     else:
#         raise TypeError(f"Unsupported type: {type(model)}")

#     strict_schema = ensure_strict_schema(json_schema)

#     return {
#         "type": "json_schema",
#         "json_schema": {
#             "schema": strict_schema,
#             "name": getattr(model, "__name__", ""),
#             "strict": True,
#         },
#     }
