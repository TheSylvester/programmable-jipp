# jipp_examples.py
# You can run this file to see examples of how to use the jipp library.
# To run the examples, you need to have a OpenAI API key.
# You can get an API key from https://platform.openai.com/account/api-keys
#
# To run the examples, you can use the following command:
# python jipp_examples.py


from jipp.examples.run_jipp_groq_examples import run_ask_groq_basic


async def main():
    await run_ask_groq_basic()
