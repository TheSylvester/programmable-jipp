import pytest
import asyncio
from nextcord.ext import commands
from task_manager import TaskManager, CreateTask, StopTask, ListTasks


class MockBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix="!")


@pytest.fixture
def bot():
    return MockBot()


@pytest.fixture
def task_manager(bot):
    return TaskManager(bot)


@pytest.mark.asyncio
async def test_create_task(task_manager):
    # Test creating a task
    task_name = "test_task"
    interval = 5
    prompt = "Test prompt"

    async def mock_function():
        pass

    result = task_manager.create_task(task_name, interval, mock_function, prompt=prompt)
    assert (
        f"Task '{task_name}' created and scheduled to start with an interval of {interval} minutes."
        in result
    )
    assert task_name in task_manager.jobs
    assert task_name in task_manager.jobs_metadata

    # Test creating a duplicate task
    duplicate_result = task_manager.create_task(
        task_name, interval, mock_function, prompt=prompt
    )
    assert f"Task '{task_name}' already exists." in duplicate_result


@pytest.mark.asyncio
async def test_stop_task(task_manager):
    # Create a task first
    task_name = "test_stop_task"
    interval = 5

    async def mock_function():
        pass

    task_manager.create_task(task_name, interval, mock_function)

    # Verify that the task was created
    assert task_name in task_manager.jobs
    assert task_name in task_manager.jobs_metadata

    # Test stopping the task
    result = await task_manager._stop_task(task_name)
    assert f"Task '{task_name}' stopped and removed." in result
    assert task_name not in task_manager.jobs
    assert task_name not in task_manager.jobs_metadata

    # Test stopping a non-existent task
    non_existent_result = await task_manager._stop_task("non_existent_task")
    assert "Task 'non_existent_task' does not exist." in non_existent_result


@pytest.mark.asyncio
async def test_list_tasks(task_manager):
    # Create some tasks
    task_names = ["task1", "task2", "task3"]
    interval = 5

    async def mock_function():
        pass

    for task_name in task_names:
        task_manager.create_task(
            task_name, interval, mock_function, prompt="Test prompt"
        )

    # Test listing tasks
    task_list = task_manager._list_tasks()
    assert len(task_list) == len(task_names)
    for task in task_list:
        assert task["task_name"] in task_names
        assert task["interval"] == interval
        assert "job" in task

    # Stop a task and check the list again
    await task_manager._stop_task("task2")
    updated_task_list = task_manager._list_tasks()
    assert len(updated_task_list) == len(task_names) - 1
    assert all(task["task_name"] in ["task1", "task3"] for task in updated_task_list)


@pytest.mark.asyncio
async def test_task_execution(task_manager):
    task_name = "execution_test"
    interval = 1 / 60  # Run every second for testing purposes
    execution_count = 0

    async def test_function():
        nonlocal execution_count
        execution_count += 1

    task_manager.create_task(task_name, interval, test_function)

    # Wait for a few seconds to allow the task to execute multiple times
    await asyncio.sleep(3)

    assert execution_count > 0, "Task should have executed at least once"

    await task_manager._stop_task(task_name)


# Test the setup function
def test_setup(bot):
    from task_manager import setup

    setup(bot)
    assert any(isinstance(cog, TaskManager) for cog in bot.cogs.values())
