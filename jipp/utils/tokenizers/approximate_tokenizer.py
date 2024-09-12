def count_tokens_approximate(text: str) -> int:
    """
    Approximate token count using a simple heuristic of 1 token per 3 characters.
    """
    return len(text) // 3  # Estimate: average 1 token per 3 characters
