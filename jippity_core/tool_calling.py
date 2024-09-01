from typing import List, Union
from openai.types.chat import ChatCompletion
from pydantic import BaseModel
from typing import List, Dict, Type
import json

from pytest import Function

from models.llm_models import Message, Tool, ToolCall
from typing import List, Dict, Any
from openai.types.chat import ChatCompletion
from models.llm_models import Message, ToolCall, Function


# def handle_tool_calling(
#     client_input: Dict[str, Any], llm_client, available_functions: List[Dict[str, Any]]
# ) -> List[Message]:
#     messages = []
#     while True:
#         response = llm_client.generate_response(client_input)
#         messages.append(response)

#         if not response.tool_calls:
#             break

#         function_messages = process_tool_calls(response, available_functions)
#         messages.extend(function_messages)
#         client_input["messages"].extend(function_messages)

#     return messages


async def execute_tool_call(
    tool_call: ToolCall,
    tools: List[Tool],
) -> str:
    for tool in tools:
        if tool.name == tool_call.function.name:
            # Parse the arguments
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                return f"Error: Invalid JSON in function arguments for {tool_call.function.name}"

            # Execute the function
            try:
                response = await tool(**args)
                return response
            except Exception as e:
                return f"Error executing {tool_call.function.name}: {str(e)}"

    return f"Error: Function {tool_call.function.name} not found"


def create_function_response_message(
    function_output: str, tool_call_id: str, function_name: str
) -> Message:
    return Message(
        role="function",
        content=function_output,
        tool_calls=[
            ToolCall(
                id=tool_call_id,
                type="function",
                function=Function(
                    name=function_name,
                    arguments={},  # We don't need arguments in the response
                ),
            )
        ],
    )
