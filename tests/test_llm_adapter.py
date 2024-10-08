import pytest
from jipp.adapters.llm_adapter import execute_llm
from jipp.models.jipp_models import LLMError


@pytest.mark.asyncio
async def test_execute_llm():
    try:
        result = await execute_llm(model="gpt-4o-mini", prompt="Hello, world!")
        assert isinstance(result, dict)
        assert "messages" in result
        # Add more assertions based on the expected structure of the result
    except LLMError as e:
        pytest.fail(f"LLMError occurred: {str(e)}")
