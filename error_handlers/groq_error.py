from groq import APIError, RateLimitError, AuthenticationError, BadRequestError


def handle_groq_api_error(error):
    print(f"Groq API Error: {error}")


def handle_groq_rate_limit_error(error):
    print(f"Groq Rate Limit Exceeded: {error}")


def handle_groq_authentication_error(error):
    print(f"Groq Authentication Error: {error}")


def handle_groq_bad_request_error(error):
    print(f"Groq Bad Request: {error}")


def handle_groq_generic_error(error):
    print(f"An unexpected Groq error occurred: {error}")


# A dictionary to map errors to handlers
GroqErrorHandlers = {
    APIError: handle_groq_api_error,
    RateLimitError: handle_groq_rate_limit_error,
    AuthenticationError: handle_groq_authentication_error,
    BadRequestError: handle_groq_bad_request_error,
}


def handle_groq_error(error):
    """Handles Groq errors using the mapped handlers."""
    handler = GroqErrorHandlers.get(type(error), handle_groq_generic_error)
    handler(error)
