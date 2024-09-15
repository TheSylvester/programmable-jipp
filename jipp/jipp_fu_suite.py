import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import asyncio
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel
from jipp.models.jipp_models import Conversation, LLMError, Tool, NotGiven, NOT_GIVEN
from jipp.llms.llm_selector import get_model_profile, ModelProfileNotFoundError
from jipp.jipp_engine import ask_llm
from jipp.utils.logging_utils import log


async def ask_llms(
    *,
    models: List[str],
    prompt: str,
    system: Optional[str] = None,
    conversation: Optional[Conversation] = None,
    response_format: Union[type[BaseModel], NotGiven] = NOT_GIVEN,
    temperature: Union[Optional[float], NotGiven] = NOT_GIVEN,
    top_p: Union[Optional[float], NotGiven] = NOT_GIVEN,
    n: Union[Optional[int], NotGiven] = NOT_GIVEN,
    max_tokens: Union[Optional[int], NotGiven] = NOT_GIVEN,
    stop: Union[Optional[str], List[str], NotGiven] = NOT_GIVEN,
    presence_penalty: Union[Optional[float], NotGiven] = NOT_GIVEN,
    frequency_penalty: Union[Optional[float], NotGiven] = NOT_GIVEN,
    logit_bias: Union[Optional[Dict[str, int]], NotGiven] = NOT_GIVEN,
    tools: Union[
        List[Dict[str, Union[str, type[BaseModel]]] | Tool], NotGiven
    ] = NOT_GIVEN,
    tool_choice: Union[Optional[str], NotGiven] = NOT_GIVEN,
    seed: Union[Optional[int], NotGiven] = NOT_GIVEN,
    user: Union[str, NotGiven] = NOT_GIVEN,
    images: Union[List[Dict[str, str]], NotGiven] = NOT_GIVEN,
    api_key: Optional[str] = None,
    organization: Optional[str] = None,
    timeout: Union[float, NotGiven] = NOT_GIVEN,
    model_specific_args: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Conversation]:
    """
    Asynchronously interact with multiple language models to generate responses.

    Args:
        models (List[str]): List of model names to use.
        prompt (str): The user's input prompt.
        system (Optional[str]): The system message to set the context.
        conversation (Optional[Conversation]): An existing conversation to continue.
        response_format (Union[type[BaseModel], NotGiven]): The expected response format.
        temperature (Union[Optional[float], NotGiven]): Controls randomness in output.
        top_p (Union[Optional[float], NotGiven]): Controls diversity of output.
        n (Union[Optional[int], NotGiven]): Number of completions to generate.
        max_tokens (Union[Optional[int], NotGiven]): Maximum number of tokens to generate.
        stop (Union[Optional[str], List[str], NotGiven]): Sequences where the API will stop generating.
        presence_penalty (Union[Optional[float], NotGiven]): Penalizes new tokens based on their presence in the text so far.
        frequency_penalty (Union[Optional[float], NotGiven]): Penalizes new tokens based on their frequency in the text so far.
        logit_bias (Union[Optional[Dict[str, int]], NotGiven]): Modifies the likelihood of specified tokens appearing in the completion.
        tools (Union[List[Dict[str, Union[str, type[BaseModel]]] | Tool], NotGiven]): List of tools available to the model.
        tool_choice (Union[Optional[str], NotGiven]): Influences how the model chooses to call functions.
        seed (Union[Optional[int], NotGiven]): Sets a random seed for deterministic output.
        user (Union[str, NotGiven]): A unique identifier representing the end-user.
        images (Union[List[Dict[str, str]], NotGiven]): List of images to include in the prompt.
        api_key (Optional[str]): API key for authentication.
        organization (Optional[str]): Organization ID for API requests.
        timeout (Union[float, NotGiven]): Maximum time to wait for a response.
        model_specific_args (Optional[Dict[str, Dict[str, Any]]]): Model-specific argument overrides.

    Returns:
        Dict[str, Conversation]: A dictionary with model names as keys and their output Conversations as values.

    Raises:
        ValueError: If the models list is empty or contains invalid model names.
        LLMError: If there's an error in processing the language model requests.
    """
    if not models:
        raise ValueError("The 'models' list cannot be empty.")

    model_specific_args = model_specific_args or {}
    results = {}
    tasks = []

    global_args = {
        "prompt": prompt,
        "system": system,
        "conversation": conversation,
        "response_format": response_format,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "max_tokens": max_tokens,
        "stop": stop,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "logit_bias": logit_bias,
        "tools": tools,
        "tool_choice": tool_choice,
        "seed": seed,
        "user": user,
        "images": images,
        "api_key": api_key,
        "organization": organization,
        "timeout": timeout,
    }

    for model in models:
        try:
            # Validate the model
            get_model_profile(model)
        except ModelProfileNotFoundError:
            log.warning(f"Invalid model name: {model}. Skipping this model.")
            continue

        # Prepare arguments for this model
        model_args = global_args.copy()
        model_args.update(model_specific_args.get(model, {}))

        # Create a task for this model
        task = asyncio.create_task(ask_llm_wrapper(model=model, **model_args))
        tasks.append(task)

    if not tasks:
        raise ValueError("No valid models to process.")

    # Run all tasks concurrently
    completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    for model, result in zip(models, completed_tasks):
        if isinstance(result, Exception):
            log.error(f"Error occurred for model {model}: {str(result)}")
            results[model] = LLMError(f"Error in {model} API call: {str(result)}")
        else:
            results[model] = result

    return results


async def ask_llm_wrapper(model: str, **kwargs: Any) -> Conversation:
    """
    Wrapper function to handle individual ask_llm calls with proper error handling.
    """
    try:
        return await ask_llm(model=model, **kwargs)
    except Exception as e:
        log.error(f"Error in ask_llm for model {model}: {str(e)}")
        raise LLMError(f"Error in {model} API call: {str(e)}")


if __name__ == "__main__":

    async def main():
        models = ["gpt-4o-mini", "llama-3.1-8b-instant"]
        prompt = "Explain the concept of quantum computing in simple terms."
        system = "You are a helpful assistant specializing in explaining complex topics simply."

        try:
            results = await ask_llms(
                models=models,
                prompt=prompt,
                system=system,
                temperature=0.7,
                max_tokens=150,
            )

            for model, conversation in results.items():
                log.info(f"\nResponse from {model}:")
                if isinstance(conversation, LLMError):
                    log.error(f"Error: {conversation}")
                else:
                    log.info(conversation.messages[-1].content)
                    log.info(f"Tokens used: {conversation.usage.total_tokens}")

        except Exception as e:
            log.error(f"An error occurred: {str(e)}")

    asyncio.run(main())
