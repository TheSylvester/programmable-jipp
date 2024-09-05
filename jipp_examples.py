# jipp_examples.py
# You can run this file to see examples of how to use the jipp library.
# To run the examples, you need to have a OpenAI API key.
# You can get an API key from https://platform.openai.com/account/api-keys
#
# To run the examples, you can use the following command:
# python jipp_examples.py

from jipp.examples.run_jipp_examples import run_ask_llm_basic
from jipp.examples.run_jipp_openai_examples import run_ask_openai_basic


async def main():
    await run_ask_llm_basic()
    await run_ask_openai_basic()
