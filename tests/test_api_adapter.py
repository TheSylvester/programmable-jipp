import pytest
from jipp.adapters.api_adapter import execute_api, APIError


@pytest.mark.asyncio
async def test_execute_api_get_success():
    # Example 1: GET request to a public API
    result = await execute_api("https://jsonplaceholder.typicode.com/todos/1", "GET")

    assert isinstance(result, dict)
    assert "id" in result
    assert "title" in result
    assert "completed" in result


@pytest.mark.asyncio
async def test_execute_api_post_success():
    # Example 2: POST request with data
    data = {"title": "foo", "body": "bar", "userId": 1}
    result = await execute_api(
        "https://jsonplaceholder.typicode.com/posts",
        "POST",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    assert isinstance(result, dict)
    assert "id" in result
    assert result["title"] == "foo"
    assert result["body"] == "bar"
    assert result["userId"] == 1


@pytest.mark.asyncio
async def test_execute_api_error_response():
    # Example 3: GET request with error (404)
    with pytest.raises(APIError) as exc_info:
        await execute_api("https://jsonplaceholder.typicode.com/nonexistent", "GET")

    assert "API request failed" in str(exc_info.value)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_execute_api_with_params():
    # Additional test: GET request with query parameters
    params = {"userId": 1}
    result = await execute_api(
        "https://jsonplaceholder.typicode.com/posts", "GET", params=params
    )

    assert isinstance(result, list)
    assert len(result) > 0
    assert all(post["userId"] == 1 for post in result)


@pytest.mark.asyncio
async def test_execute_api_put():
    # Additional test: PUT request
    data = {"id": 1, "title": "updated title", "body": "updated body", "userId": 1}
    result = await execute_api(
        "https://jsonplaceholder.typicode.com/posts/1",
        "PUT",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    assert isinstance(result, dict)
    assert result["id"] == 1
    assert result["title"] == "updated title"
    assert result["body"] == "updated body"


@pytest.mark.asyncio
async def test_execute_api_delete():
    # Additional test: DELETE request
    result = await execute_api("https://jsonplaceholder.typicode.com/posts/1", "DELETE")

    assert result == {}  # JSONPlaceholder returns an empty object for DELETE


@pytest.mark.asyncio
async def test_execute_api_timeout():
    # Test timeout scenario
    with pytest.raises(APIError) as exc_info:
        await execute_api(
            "https://httpbin.org/delay/5",  # This endpoint delays response by 5 seconds
            "GET",
            timeout=1.0,  # Set a short timeout
        )

    assert "Request timed out" in str(exc_info.value)
