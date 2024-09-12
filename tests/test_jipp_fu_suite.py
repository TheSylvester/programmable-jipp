import pytest
from jipp.jipp_fu_suite import ask_llms
from jipp.models.jipp_models import Conversation, LLMError, CompletionUsage
from typing import Dict, Union


@pytest.mark.asyncio
async def test_ask_llms_basic():
    models = ["gpt-4o-mini", "llama-3.1-8b-instant"]
    prompt = "Say 'Hello, World!'"
    system = "You are a helpful assistant."

    results = await ask_llms(models=models, prompt=prompt, system=system)

    assert isinstance(results, dict)
    assert len(results) == len(models)
    for model, conversation in results.items():
        assert model in models
        assert isinstance(conversation, Conversation)
        # Check if the response contains either 'hello' or the original prompt
        assert (
            "hello" in str(conversation).lower()
            or prompt.lower() in str(conversation).lower()
        )
        assert conversation.usage is not None
        assert model in conversation.model
        assert conversation.finish_reason in ["stop", "length"]


@pytest.mark.asyncio
async def test_ask_llms_with_error():
    models = ["gpt-4o-mini", "non-existent-model"]
    prompt = "This should partially fail"

    results = await ask_llms(models=models, prompt=prompt)

    assert isinstance(results, dict)
    # Only valid models should be in the results
    assert len(results) == 1
    assert "gpt-4o-mini" in results
    assert isinstance(results["gpt-4o-mini"], Conversation)
    # The non-existent model should not be in the results
    assert "non-existent-model" not in results


@pytest.mark.asyncio
async def test_ask_llms_with_model_specific_args():
    models = ["gpt-4o-mini", "llama-3.1-8b-instant"]
    prompt = "Generate a random number"
    model_specific_args = {
        "gpt-4o-mini": {"temperature": 0.7},
        "llama-3.1-8b-instant": {"temperature": 0.9},
    }

    results = await ask_llms(
        models=models, prompt=prompt, model_specific_args=model_specific_args
    )

    assert isinstance(results, dict)
    assert len(results) == len(models)
    for model, conversation in results.items():
        assert model in models
        assert isinstance(conversation, Conversation)
        assert "random number" in str(conversation).lower()


@pytest.mark.asyncio
async def test_ask_llms_empty_models_list():
    with pytest.raises(ValueError) as exc_info:
        await ask_llms(models=[], prompt="This should fail")

    assert "The 'models' list cannot be empty." in str(exc_info.value)


@pytest.mark.asyncio
async def test_ask_llms_all_invalid_models():
    models = ["invalid-model-1", "invalid-model-2"]
    prompt = "This should fail"

    with pytest.raises(ValueError) as exc_info:
        await ask_llms(models=models, prompt=prompt)

    assert "No valid models to process." in str(exc_info.value)
