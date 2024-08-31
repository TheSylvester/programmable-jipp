from openai import (
    APIError,
    RateLimitError,
    AuthenticationError,
    BadRequestError,  # Updated from InvalidRequestError
    Timeout,
)


def handle_openai_api_error(error):
    print(f"OpenAI API Error: {error}")


def handle_openai_rate_limit_error(error):
    print(f"OpenAI Rate Limit Exceeded: {error}")


def handle_openai_authentication_error(error):
    print(f"OpenAI Authentication Error: {error}")


def handle_openai_bad_request_error(error):
    print(f"OpenAI Bad Request: {error}")


def handle_openai_service_unavailable_error(error):
    print(f"OpenAI Service Unavailable: {error}")


def handle_openai_timeout_error(error):
    print(f"OpenAI Request Timeout: {error}")


def handle_openai_generic_error(error):
    print(f"An unexpected OpenAI error occurred: {error}")


# A dictionary to map errors to handlers
OpenAIErrorHandlers = {
    APIError: handle_openai_api_error,
    RateLimitError: handle_openai_rate_limit_error,
    AuthenticationError: handle_openai_authentication_error,
    BadRequestError: handle_openai_bad_request_error,  # Updated from InvalidRequestError
    Timeout: handle_openai_timeout_error,
}


def handle_openai_error(error):
    """Handles OpenAI errors using the mapped handlers."""
    handler = OpenAIErrorHandlers.get(type(error), handle_openai_generic_error)
    handler(error)
