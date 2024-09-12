import base64
import io
import logging
from PIL import Image
from jinja2 import Template
from jipp.llms.llm_selector import get_model_profile
from jipp.models.jipp_models import (
    Conversation,
    ImageURL,
    LLMMessage,
    LLMError,
    MessageContentImage,
    MessageContentText,
    Tool,
    ToolCall,
)
from typing import Literal, Optional, List, Dict, Type, Union
from pydantic import BaseModel
from jipp.models.jipp_models import NotGiven, NOT_GIVEN


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant"
DEFAULT_MODEL = "llama-3.1-8b-instant"


async def ask_llm(
    *,
    model: str,
    prompt: str,
    system: Optional[str] = None,
    conversation: Optional[Conversation] = None,
    response_format: type[BaseModel] | NotGiven = NOT_GIVEN,
    temperature: Optional[float] | NotGiven = NOT_GIVEN,
    top_p: Optional[float] | NotGiven = NOT_GIVEN,
    n: Optional[int] | NotGiven = NOT_GIVEN,
    max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
    stop: Union[Optional[str], List[str]] | NotGiven = NOT_GIVEN,
    presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
    frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
    logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
    tools: List[Dict[str, str | Type[BaseModel]] | Tool] | NotGiven = NOT_GIVEN,
    tool_choice: Optional[str] | NotGiven = NOT_GIVEN,
    seed: Optional[int] | NotGiven = NOT_GIVEN,
    user: str | NotGiven = NOT_GIVEN,
    images: List[Dict[Literal["url", "filepath"], str]] | NotGiven = NOT_GIVEN,
    api_key: Optional[str] = None,
    organization: Optional[str] = None,
    timeout: float | NotGiven = NOT_GIVEN,
    **variables,
) -> Conversation:
    """
    Asynchronously interact with a language model to generate a response.

    Args:
        model (str): The name of the language model to use.
        prompt (str): The user's input prompt.
        system (Optional[str]): The system message to set the context.
        conversation (Optional[Conversation]): An existing conversation to continue.
        response_format (type[BaseModel] | NotGiven): The expected response format.
        temperature (Optional[float] | NotGiven): Controls randomness in output.
        top_p (Optional[float] | NotGiven): Controls diversity of output.
        n (Optional[int] | NotGiven): Number of completions to generate.
        max_tokens (Optional[int] | NotGiven): Maximum number of tokens to generate.
        stop (Union[Optional[str], List[str]] | NotGiven): Sequences where the API will stop generating.
        presence_penalty (Optional[float] | NotGiven): Penalizes new tokens based on their presence in the text so far.
        frequency_penalty (Optional[float] | NotGiven): Penalizes new tokens based on their frequency in the text so far.
        logit_bias (Optional[Dict[str, int]] | NotGiven): Modifies the likelihood of specified tokens appearing in the completion.
        tools (List[Dict[str, str | Type[BaseModel]] | Tool] | NotGiven): List of tools available to the model.
        tool_choice (Optional[str] | NotGiven): Influences how the model chooses to call functions.
        seed (Optional[int] | NotGiven): Sets a random seed for deterministic output.
        user (str | NotGiven): A unique identifier representing the end-user.
        images (List[Dict[Literal["url", "filepath"], str]] | NotGiven): List of images to include in the prompt.
        api_key (Optional[str]): API key for authentication.
        organization (Optional[str]): Organization ID for API requests.
        timeout (float | NotGiven): Maximum time to wait for a response.
        **variables: Additional variables to be used in prompt rendering.

    Returns:
        Conversation: The resulting conversation including the model's response.

    Raises:
        LLMError: If there's an error in processing the language model request.
    """

    kwargs = {
        "model": model,
        "response_format": response_format,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "max_tokens": max_tokens,
        "stop": stop,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "logit_bias": logit_bias,
        "tool_choice": tool_choice,
        "seed": seed,
        "user": user,
        "timeout": timeout,
    }

    # Convert tools to Tools if any
    if tools is not NOT_GIVEN:
        tools = convert_tool_dicts(tools)
        kwargs["tools"] = [tool.schema for tool in tools]

    # Remove NOT_GIVEN values
    kwargs = {k: v for k, v in kwargs.items() if v is not NOT_GIVEN}

    try:
        # get the model profile and client
        model_profile = get_model_profile(model)
        ask_client = model_profile.client

        # start or retrieve old conversation
        messages = [m for m in conversation.messages] if conversation else []

        # system prompt
        if system:
            messages = set_system_message(system, variables, messages)

        # user prompt + images
        rendered_user_prompt = render_template(prompt, **variables)
        content = rendered_user_prompt
        if images is not NOT_GIVEN:
            content = add_images(images, rendered_user_prompt)
        user_message = LLMMessage(role="user", content=content)
        messages.append(user_message)

        # call LLM
        print(f"Prepared messages: {messages}\nGenerating response")
        response = await ask_client(messages=messages, **kwargs)
        print(f"Got response: {response}")

        # handle tool call requests until LLM stops calling tools
        while response.message.tool_calls:
            tool_calls = response.message.tool_calls
            print(f"Got tool calls: {tool_calls}")
            # append the tool call request back into the messages
            messages.append(response.message)

            for tool_call in tool_calls:
                tool_message = await handle_tool_call_request(tool_call, tools)
                messages.append(tool_message)

            response = await ask_client(messages=messages, **kwargs)
            print(f"Got response after tool calls: {response}")

        # parse output if we have a response model
        add_parsed = {}
        if response_format is not NOT_GIVEN and (
            parsed := parse(response_format, response)
        ):
            add_parsed["parsed"] = parsed

        return Conversation(
            messages=messages + [response.message],
            usage=response.usage,
            model=response.model,
            finish_reason=response.finish_reason,
            **add_parsed,
        )

    except Exception as e:
        # Change this part to always raise LLMError
        raise LLMError(f"An error occurred: {str(e)}")


def parse(response_format, response) -> Optional[BaseModel]:
    """
    Parse the response using the response_format.

    Args:
        response_format (Type[BaseModel]): The expected format of the response.
        response (LLMResponse): The response from the language model.

    Returns:
        Optional[BaseModel]: The parsed response if successful, otherwise None.
    """
    try:
        if issubclass(response_format, BaseModel):
            return response_format.model_validate_json(response.message.content)
    except Exception as e:
        print(
            f"WARNING: Parsing error from output to response_format: {e} - output format IGNORED"
        )

    return None


def add_images(
    images: List[Dict[Literal["url", "filepath"], str]], rendered_user_prompt: str
) -> List[Union[MessageContentText, MessageContentImage]]:
    """
    Add images to the user message content.

    Args:
        images (List[Dict[Literal["url", "filepath"], str]]): List of image data.
        rendered_user_prompt (str): The rendered user prompt.

    Returns:
        List[Union[MessageContentText, MessageContentImage]]: The combined text and image content.
    """
    user_message_content = [
        MessageContentText(type="text", text=rendered_user_prompt)
    ] + process_image_inputs_into_contents(images)

    return user_message_content


def convert_tool_dicts(
    tools: List[Dict[str, str | Type[BaseModel]] | Tool]
) -> list[Tool]:
    """
    Convert a list of tool dictionaries or Tool objects into a list of Tool objects.

    Args:
        tools (List[Dict[str, str | Type[BaseModel]] | Tool]): List of tool data.

    Returns:
        list[Tool]: List of Tool objects.

    Raises:
        ValueError: If an invalid tool format is provided.
    """
    tools_models: list[Tool] = []
    for tool in tools:
        if isinstance(tool, Tool):
            tools_models.append(tool)
        elif isinstance(tool, dict):
            tools_models.append(Tool(**tool))
        else:
            raise ValueError(f"Invalid tool: {tool}")

    return tools_models


def set_system_message(
    system: str, variables: dict, messages: List[LLMMessage]
) -> List[LLMMessage]:
    """
    Add or replace the system message in the conversation.

    Args:
        system (str): The system message template.
        variables (dict): Variables for template rendering.
        messages (List[LLMMessage]): Existing conversation messages.

    Returns:
        List[LLMMessage]: Updated list of messages with the new system message.
    """
    if system is None or system == "":
        return messages

    system_prompt = render_template(system, **variables)
    system_message = LLMMessage(role="system", content=system_prompt)

    # remove the first message if it's a system message
    if messages and messages[0].role == "system":
        messages = messages[1:]
    messages = [system_message] + messages
    return messages


def render_template(template_string: str, **kwargs) -> str:
    """
    Render a Jinja2 template with the provided variables.

    Args:
        template_string (str): The template string to render.
        **kwargs: Variables to use in template rendering.

    Returns:
        str: The rendered template string.
    """
    template = Template(template_string)
    return template.render(**kwargs)


def process_image_inputs_into_contents(
    images: List[Dict[Literal["url", "filepath"], str]]
) -> List[MessageContentImage]:
    """
    Process a list of image inputs into MessageContentImage objects.

    Args:
        images (List[Dict[Literal["url", "filepath"], str]]): List of image data.

    Returns:
        List[MessageContentImage]: List of processed image contents.

    Raises:
        ValueError: If an invalid image input is provided.
    """
    user_message_content = []
    for image in images:
        if isinstance(image, str):
            content = _create_image_message_content_from_url(image)
        elif "url" in image:
            content = _create_image_message_content_from_url(image["url"])
        elif "filepath" in image:
            content = _create_image_message_content_from_filepath(image["filepath"])
        else:
            raise ValueError(f"Invalid image input: {image}")
        user_message_content.append(content)

    return user_message_content


def _create_image_message_content_from_url(url: str) -> MessageContentImage:
    """
    Create a MessageContentImage object from a URL.

    Args:
        url (str): The URL of the image.

    Returns:
        MessageContentImage: The created image content object.
    """
    return MessageContentImage(
        type="image_url", image_url=ImageURL(url=url, detail="auto")
    )


def _create_image_message_content_from_filepath(
    filepath: str,
) -> MessageContentImage:
    """
    Create a MessageContentImage object from a file path.

    Args:
        filepath (str): The file path of the image.

    Returns:
        MessageContentImage: The created image content object.
    """
    with open(filepath, "rb") as image_file:
        image_bytes = io.BytesIO(image_file.read())
        image = Image.open(image_bytes)
        format = image.format
        base64_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
        url = f"data:image/{format.lower()};base64,{base64_image}"

        return MessageContentImage(
            type="image_url", image_url=ImageURL(url=url, detail="auto")
        )


async def handle_tool_call_request(
    tool_call: ToolCall, tools: List[Tool]
) -> LLMMessage:
    """
    Handle a tool call request and return the tool message.

    Args:
        tool_call (ToolCall): The tool call request.
        tools (List[Tool]): List of available tools.

    Returns:
        LLMMessage: The tool message to return to the LLM.
    """
    tool_response = await execute_tool_call(tool_call, tools)
    print(f"Tool response: {tool_response}")
    tool_message = LLMMessage(
        tool_call_id=tool_call.id, role="tool", content=tool_response
    )
    return tool_message


async def execute_tool_call(tool_call: ToolCall, tools: List[Tool]) -> str:
    """
    Execute a tool call selected from the list of tools.

    Args:
        tool_call (ToolCall): The tool call to execute.
        tools (List[Tool]): List of available tools.

    Returns:
        str: The result of the tool execution.

    Raises:
        ValueError: If the requested tool is not found.
    """
    logging.debug(tool_call)
    for tool in tools:
        logging.debug(tool)
        if tool.schema.__name__ == tool_call.function.name:
            validated_args = tool.schema.model_validate_json(
                tool_call.function.arguments
            )
            arguments = validated_args.model_dump()

            logging.debug(arguments)
            result = await tool(**arguments)
            return result

    raise ValueError(f"Tool {tool_call.function.name} not found")
