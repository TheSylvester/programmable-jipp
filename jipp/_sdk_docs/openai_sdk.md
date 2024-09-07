# OpenAI SDK

## Chat Completions

The chat completion endpoint (`POST https://api.openai.com/v1/chat/completions`) is used to generate a completion based on a conversation. You can provide a series of messages as input, and the model will return a message as output.

### Request

The request body includes the following parameters:

- **model**: _string_ (required)  
  ID of the model to use. You can use the `gpt-4`, `gpt-4-turbo`, or `gpt-3.5-turbo` model family.

- **messages**: _array_ of _message objects_ (required)  
  A list of messages in the conversation. Each message must contain the following fields:

  - **role**: _string_ (required)  
    The role of the message author. One of `system`, `user`, `assistant`, or `function`.
  - **content**: _string_ (required unless `role` is `function`)  
    The content of the message. Optional if `role` is `function`.
  - **name**: _string_ (optional)  
    Name of the author of this message. Required if `role` is `function`.

- **temperature**: _number_ (optional, default: 1)  
  What sampling temperature to use. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.

- **top_p**: _number_ (optional, default: 1)  
  An alternative to sampling with temperature, called nucleus sampling. This controls the diversity of the output. 1 means the model considers all options, while lower values like 0.1 mean the model only considers the top options.

- **n**: _integer_ (optional, default: 1)  
  How many chat completion choices to generate for each input message.

- **stream**: _boolean_ (optional, default: false)  
  If `true`, partial message deltas will be sent as data-only `server-sent events` as they become available. The stream will terminate with a `data: [DONE]` message.

- **stop**: _string_ or _array_ of _strings_ (optional)  
  Up to 4 sequences where the API will stop generating further tokens.

- **max_tokens**: _integer_ (optional)  
  The maximum number of tokens allowed for the generated answer. By default, the number of tokens the model can return will be limited by the model’s context length.

- **presence_penalty**: _number_ (optional, default: 0)  
  Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.

- **frequency_penalty**: _number_ (optional, default: 0)  
  Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, reducing the model's likelihood to repeat the same line verbatim.

- **logit_bias**: _map_ (optional)  
  Modify the likelihood of specified tokens appearing in the completion.

- **user**: _string_ (optional)  
  A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.

- **functions**: _array_ of _function definitions_ (optional)  
  A list of functions the model may use, including the function name and parameters. This allows the model to generate code that calls the function.

- **function_call**: _string_ or _object_ (optional)  
  Controls how the model calls a function:
  - `"none"` means the model does not call a function.
  - `"auto"` means the model can decide to call a function if it believes one is necessary.
  - A `dictionary` can be used to specify a particular function to call, along with optional arguments.

### Example Usage

Here are some Python examples for how to use the Chat Completion endpoint:

#### Basic Chat Completion

```python
import openai

response = openai.ChatCompletion.create(
  model="gpt-4",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"}
  ]
)

print(response.choices[0].message['content'])
```

#### Image Generation with Chat Completion

You can include image URLs in the conversation and have the model respond with text based on the images.

```python
import openai

response = openai.ChatCompletion.create(
  model="gpt-4",
  messages=[
    {"role": "system", "content": "You are a creative assistant."},
    {"role": "user", "content": "Describe this image: https://example.com/image.jpg"}
  ]
)

print(response.choices[0].message['content'])
```

#### Function Calling / Tool Use

You can define functions and allow the model to call them as needed.

```python
import openai

response = openai.ChatCompletion.create(
  model="gpt-4",
  messages=[
    {"role": "user", "content": "Calculate the sum of 15 and 25."}
  ],
  functions=[
    {
      "name": "add_numbers",
      "description": "Adds two numbers together.",
      "parameters": {
        "type": "object",
        "properties": {
          "a": {"type": "number", "description": "First number"},
          "b": {"type": "number", "description": "Second number"}
        },
        "required": ["a", "b"]
      }
    }
  ],
  function_call={"name": "add_numbers"}
)

print(response['choices'][0]['message']['function_call'])
```

In this example, the model is able to call the `add_numbers` function with the specified parameters.

---

This information should serve as detailed documentation for an LLM to use when writing critical code related to chat completions, image handling, and function/tool calling using the OpenAI API.

The example code you provided is indeed a correct and updated version of how to implement streaming in Python using OpenAI's `gpt-4o-mini` model. It allows you to stream responses by setting the `stream` parameter to `true`. Below is a detailed breakdown:

### Example Code for Streaming

```python
from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  stream=True  # Enable streaming
)

for chunk in completion:
  print(chunk.choices[0].delta)
```

### Expected Response

When you run this code, the API will return chunks of the message as they are generated. The response format for each chunk looks like this:

```json
{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4o-mini", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}

{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4o-mini", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{"content":"Hello"},"logprobs":null,"finish_reason":null}]}

....

{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4o-mini", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"stop"}]}
```

### Key Details

- **`stream=True`**: This enables streaming mode, where the response is broken into chunks as the model generates text.
- **Response chunks**: Each chunk contains a partial piece of the completion, which you can print or process in real-time.
- **Delta**: The `delta` key within the `choices` array contains the partial content that is being generated.
- **Finish Reason**: The final chunk contains a `finish_reason` key that indicates why the completion ended (e.g., `"stop"` when the model finishes normally).

This example aligns with the streaming feature documentation on OpenAI's platform【8†source】【9†source】.
