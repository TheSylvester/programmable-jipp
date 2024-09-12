from transformers import AutoTokenizer


def count_tokens_transformers(text: str, model: str) -> int:

    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokens = tokenizer.encode(text)
    except Exception as e:
        print(f"Error encoding text: {e}")
        return 0

    return len(tokens)
