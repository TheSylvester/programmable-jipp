import pytest
import asyncio
import os
from jippity_core.chat_llm import (
    execute_tool,
    run_code,
    create_file,
    read_file,
    execute_code,
    run_pytest_against_code,
)


@pytest.mark.asyncio
async def test_create_and_read_file():
    file_path = "pytest_generated_file.py"
    file_content = "print('Test File')"

    try:
        # Test creating a file
        create_file_result = await execute_tool(
            "create_file", {"path": file_path, "content": file_content}
        )
        assert not create_file_result["is_error"]
        assert create_file_result["content"] is True

        # Test reading the created file
        read_file_result = await execute_tool("read_file", {"path": file_path})
        assert not read_file_result["is_error"]
        assert read_file_result["content"] == file_content

    finally:
        # Cleanup: Remove the created file
        if os.path.exists(file_path):
            os.remove(file_path)


@pytest.mark.asyncio
async def test_execute_code():
    code = "print('Running Code')"
    temp_file = "temp_code_file.py"

    try:
        # Create a temporary file for the code execution
        create_file_result = await execute_tool(
            "create_file", {"path": temp_file, "content": code}
        )
        assert not create_file_result["is_error"]

        # Test executing the code
        execute_code_result = await execute_tool("execute_code", {"code": code})
        assert not execute_code_result["is_error"]
        assert "Running Code" in execute_code_result["content"]["stdout"]

    finally:
        # Cleanup: Remove the temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)


@pytest.mark.asyncio
async def test_stop_process():
    code = "import time\nfor i in range(5):\n    time.sleep(2)\n    print(f'Iteration {i}')"

    try:
        # Execute the long-running code
        execute_code_result = await execute_tool("execute_code", {"code": code})

        if (
            "process_id" in execute_code_result["content"]
            and execute_code_result["content"]["return_code"] is None
        ):
            process_id = execute_code_result["content"]["process_id"]

            # Stop the process
            stop_process_result = await execute_tool(
                "stop_process", {"process_id": process_id}
            )
            assert not stop_process_result["is_error"]
            assert stop_process_result["content"] is True
        else:
            pytest.skip("No long-running process to stop.")

    finally:
        # Optionally ensure that all processes are stopped
        if "process_id" in locals() and process_id in execute_code_result["content"]:
            await execute_tool("stop_process", {"process_id": process_id})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code, expected_output",
    [
        ("print('Hello, World!')", "Hello, World!"),
        ("x = 5\nprint(x)", "5"),
        ("for i in range(3): print(i)", "0\n1\n2"),
    ],
)
async def test_run_code(code, expected_output):
    result = await run_code(code)
    assert not result["is_error"]
    # Normalize line endings to Unix style before asserting
    stdout = result["content"]["stdout"].replace("\r\n", "\n")
    assert expected_output in stdout


@pytest.mark.asyncio
async def test_run_pytest_against_code():
    # Define the code to be tested and the pytest suite together
    combined_code = """
def add(a, b):
    return a + b

def test_add():
    assert add(1, 2) == 3
    assert add(0, 0) == 0
    assert add(-1, 1) == 0
    """

    # Run pytest against the combined code using the run_pytest_against_code function
    result = await run_pytest_against_code(combined_code, combined_code)

    # Print the stdout and stderr for debugging
    print("Pytest stdout:\n", result["stdout"])
    print("Pytest stderr:\n", result["stderr"])

    # Check that pytest ran successfully
    assert result["return_code"] == 0, "Pytest did not run successfully"
    assert "1 passed" in result["stdout"], "Not all tests passed"


if __name__ == "__main__":
    pytest.main()
