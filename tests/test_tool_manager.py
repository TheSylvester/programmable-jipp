import pytest
from nextcord.ext import commands
from jipp.models.jipp_models import Tool
from task_manager import TaskManager, CreateTask, StopTask, ListTasks
from tool_manager import ToolManager


class MockBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix="!")


@pytest.fixture
def bot():
    return MockBot()


@pytest.fixture
def tool_manager(bot):
    return ToolManager(bot)


def test_register_tool(tool_manager):
    # Create a mock tool
    mock_tool = Tool(schema=CreateTask, function=lambda: None)
    tool_manager.register_tool(mock_tool)

    assert "CreateTask" in tool_manager.tools
    assert tool_manager.tools["CreateTask"] == mock_tool


def test_register_tools(tool_manager):
    # Create mock tools
    mock_tools = [
        Tool(schema=CreateTask, function=lambda: None),
        Tool(schema=StopTask, function=lambda: None),
        Tool(schema=ListTasks, function=lambda: None),
    ]
    tool_manager.register_tools(mock_tools)

    assert set(tool_manager.tools.keys()) == set(
        ["CreateTask", "StopTask", "ListTasks"]
    )
    assert all(isinstance(tool, Tool) for tool in tool_manager.tools.values())


def test_register_available_tools(tool_manager, bot):
    # First registration should succeed
    tool_manager.register_available_tools(bot)

    # Second registration should not raise an exception
    tool_manager.register_available_tools(bot)

    assert isinstance(tool_manager.task_manager, TaskManager)
    assert set(tool_manager.tools.keys()) == set(
        ["CreateTask", "StopTask", "ListTasks"]
    )
    assert all(isinstance(tool, Tool) for tool in tool_manager.tools.values())

    # Verify that the TaskManager cog is loaded only once
    assert len(bot.cogs) == 1
    assert "TaskManager" in bot.cogs


def test_get_tool(tool_manager):
    # Register a mock tool
    mock_tool = Tool(schema=CreateTask, function=lambda: None)
    tool_manager.register_tool(mock_tool)

    retrieved_tool = tool_manager.get_tool("CreateTask")
    assert retrieved_tool == mock_tool

    non_existent_tool = tool_manager.get_tool("NonExistentTool")
    assert non_existent_tool is None


def test_get_tools(tool_manager):
    # Register mock tools
    mock_tools = [
        Tool(schema=CreateTask, function=lambda: None),
        Tool(schema=StopTask, function=lambda: None),
        Tool(schema=ListTasks, function=lambda: None),
    ]
    tool_manager.register_tools(mock_tools)

    retrieved_tools = tool_manager.get_tools(
        ["CreateTask", "StopTask", "NonExistentTool"]
    )
    assert len(retrieved_tools) == 2
    assert all(isinstance(tool, Tool) for tool in retrieved_tools)
    assert set(tool.schema.__name__ for tool in retrieved_tools) == set(
        ["CreateTask", "StopTask"]
    )


def test_integration_with_task_manager(tool_manager, bot):
    tool_manager.register_available_tools(bot)

    create_task_tool = tool_manager.get_tool("CreateTask")
    assert create_task_tool is not None
    assert create_task_tool.schema == CreateTask
    assert create_task_tool.function == tool_manager.task_manager.create_task

    stop_task_tool = tool_manager.get_tool("StopTask")
    assert stop_task_tool is not None
    assert stop_task_tool.schema == StopTask
    assert stop_task_tool.function == tool_manager.task_manager.stop_task

    list_tasks_tool = tool_manager.get_tool("ListTasks")
    assert list_tasks_tool is not None
    assert list_tasks_tool.schema == ListTasks
    assert list_tasks_tool.function == tool_manager.task_manager.list_tasks
