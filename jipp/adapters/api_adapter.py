import os
import sys
import asyncio
import aiohttp
from typing import Optional, Dict, Any

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)


class APIError(Exception):
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response: Optional[str] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)


async def execute_api(url: str, method: str = "GET", **kwargs) -> Dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        async with session.request(method, url, **kwargs) as response:
            response.raise_for_status()
            return await response.json()


async def sample_api_calls():
    # Example 1: GET request to a public API
    try:
        result = await execute_api(
            "https://jsonplaceholder.typicode.com/todos/1", "GET"
        )
        print("Example 1 - GET request:")
        print(result)
    except APIError as e:
        print(f"Error in Example 1: {e}")

    # Example 2: POST request with data
    try:
        data = {"title": "foo", "body": "bar", "userId": 1}
        result = await execute_api(
            "https://jsonplaceholder.typicode.com/posts",
            "POST",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        print("\nExample 2 - POST request:")
        print(result)
    except APIError as e:
        print(f"Error in Example 2: {e}")

    # Example 3: GET request with error (404)
    try:
        await execute_api("https://jsonplaceholder.typicode.com/nonexistent", "GET")
    except APIError as e:
        print("\nExample 3 - Error handling:")
        print(f"Caught expected error: {e}")


def main():
    asyncio.run(sample_api_calls())


if __name__ == "__main__":
    main()
