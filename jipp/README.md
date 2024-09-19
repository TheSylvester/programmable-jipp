# JIPP: Just Intelligent Prompt Passing

JIPP is a Python library designed to streamline interactions with Large Language Models (LLMs) through intelligent prompt engineering and management. It provides a unified interface for working with multiple LLM providers and offers advanced features for prompt generation and handling.

## Key Features

1. **Unified API for Multiple LLM Providers**: Seamlessly work with OpenAI, Anthropic, and Groq models.
2. **Dynamic Prompt Generation with Jinja2**: Craft complex, adaptive prompts using Jinja2 templating.
3. **Asynchronous by Design**: Leverage `asyncio` for non-blocking LLM interactions.
4. **Advanced LLM Capabilities**: Incorporate tool usage, structured outputs, and multi-modal inputs.
5. **Efficient Token Management**: Optimize token usage with built-in counting and context window management.
6. **Multi-Model Queries**: Compare responses from multiple models with a single call.
7. **Type Safety with Pydantic**: Improve code quality with extensive type hinting.
8. **Customizable Logging**: Fine-tune observability with a flexible logging system.

## Installation

Currently, JIPP is not available via pip. To use JIPP, clone the repository:

```bash
git clone https://github.com/TheSylvester/programmable-jipp.git
cd programmable-jipp
```

## Usage

Here's a basic example of how to use JIPP:

```python
import asyncio
from jipp import ask_llm

async def main():
    response = await ask_llm(
        model="gpt-4o-mini",
        prompt="Explain quantum computing in simple terms.",
        system="You are a helpful assistant specializing in explaining complex topics simply."
    )
    print(response)

asyncio.run(main())
```

## Multi-Model Queries

JIPP allows you to query multiple models simultaneously:

```python
from jipp import ask_llms

async def compare_models():
    results = await ask_llms(
        models=["gpt-4o-mini", "claude-haiku", "llama-small"],
        prompt="What are the ethical implications of AI?",
        system="You are an AI ethics expert."
    )
    for model, response in results.items():
        print(f"\n{model} response:")
        print(response)

asyncio.run(compare_models())
```

## Advanced Usage: Jippity AI

The `jippity_ai/` directory provides an example of how to build more complex AI systems using JIPP. It demonstrates:

1. A core interface (`jippity_core.py`) for managing AI interactions.
2. Modular AI programs in the `programs/` directory.
3. Separation of prompts and code using Markdown files.

While this is not part of the JIPP package itself, it serves as a valuable reference for structuring larger AI projects with JIPP.

## Project Status

JIPP is currently a private research project and is not open for public use or contributions at this time. We plan to open-source it in the future.

## License

This project is currently under private development and has no public license. All rights are reserved.

---

JIPP is designed to simplify LLM interactions and empower developers to create sophisticated AI-enhanced applications. As we continue to develop and refine JIPP, we look forward to eventually sharing it with the wider AI development community.
