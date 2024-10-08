import pytest
import asyncio
from jipp.flow.async_executor import AsyncExecutor
from jipp.flow.node_manager import NodeManager
from jipp.flow.flow_utils import Graph
from jipp.flow.node import Node, NodeDefinition
from jipp.flow.execution_context import ExecutionContext
from unittest.mock import Mock, patch


@pytest.fixture
def mock_graph():
    return Mock(spec=Graph)


@pytest.fixture
def mock_node_manager():
    return Mock(spec=NodeManager)


@pytest.fixture
def async_executor(mock_graph, mock_node_manager):
    return AsyncExecutor(mock_graph, mock_node_manager)


@pytest.mark.asyncio
async def test_execute_dag_flow(async_executor, mock_graph, mock_node_manager):
    mock_graph.flow_type = "DAG"
    mock_graph.nodes = {
        "node1": Node(
            definition=NodeDefinition(
                name="node1",
                type="Python",
                function="test_function",
                inputs={},
                outputs={},
            ),
            callable=Mock(),
        ),
        "node2": Node(
            definition=NodeDefinition(
                name="node2",
                type="Python",
                function="test_function",
                inputs={},
                outputs={},
            ),
            callable=Mock(),
        ),
    }
    mock_graph.dependencies = {"node1": [], "node2": ["node1"]}

    mock_node_manager.execute_node.side_effect = [
        {"result": "node1_output"},
        {"result": "node2_output"},
    ]

    result = await async_executor.execute({"input": "test"})

    assert isinstance(result, ExecutionContext)
    assert mock_node_manager.execute_node.call_count == 2
    assert result.get_value("${node1.result}") == "node1_output"
    assert result.get_value("${node2.result}") == "node2_output"


@pytest.mark.asyncio
async def test_execute_flex_flow(async_executor, mock_graph, mock_node_manager):
    mock_graph.flow_type = "Flex"
    mock_graph.nodes = {
        "node1": Node(
            definition=NodeDefinition(
                name="node1",
                type="Python",
                function="test_function",
                inputs={},
                outputs={},
            ),
            callable=Mock(),
        ),
    }
    mock_graph.flow_steps = [
        {"when": "${input} == 'test'", "then": ["node1"]},
        {"output": {"result": "${node1.result}"}},
    ]

    mock_node_manager.execute_node.return_value = {"result": "node1_output"}

    result = await async_executor.execute({"input": "test"})

    print(f"Execution context after execution: {result.context}")
    print(f"Execution outputs: {result.outputs}")
    print(
        f"Node manager execute_node call count: {mock_node_manager.execute_node.call_count}"
    )

    assert isinstance(result, ExecutionContext)
    assert (
        mock_node_manager.execute_node.call_count == 1
    ), "Node should be executed once"
    assert result.context.get("node1") == {
        "result": "node1_output"
    }, "Node output should be in context"
    assert result.outputs == {
        "result": "node1_output"
    }, "Flow output should match node output"

    # Test with a condition that doesn't match
    mock_node_manager.execute_node.reset_mock()
    result = await async_executor.execute({"input": "not_test"})

    print(f"Execution context after non-matching condition: {result.context}")
    print(f"Execution outputs after non-matching condition: {result.outputs}")
    print(
        f"Node manager execute_node call count after non-matching condition: {mock_node_manager.execute_node.call_count}"
    )

    assert isinstance(result, ExecutionContext)
    assert mock_node_manager.execute_node.call_count == 0, "Node should not be executed"
    assert "node1" not in result.context, "Node output should not be in context"
    assert result.outputs == {}, "Flow output should be empty"


@pytest.mark.asyncio
async def test_node_retry(async_executor, mock_graph, mock_node_manager):
    mock_graph.flow_type = "DAG"
    mock_node = Node(
        definition=NodeDefinition(
            name="retry_node",
            type="Python",
            function="test_function",
            inputs={},
            outputs={},
            max_retries=2,
        ),
        callable=Mock(),
    )
    mock_graph.nodes = {"retry_node": mock_node}
    mock_graph.dependencies = {"retry_node": []}

    mock_node_manager.execute_node.side_effect = [
        Exception("Retry error"),
        Exception("Retry error"),
        {"result": "success"},
    ]

    result = await async_executor.execute({})

    assert isinstance(result, ExecutionContext)
    assert mock_node_manager.execute_node.call_count == 3
    assert result.get_value("${retry_node.result}") == "success"


@pytest.mark.asyncio
async def test_parallel_execution(async_executor, mock_graph, mock_node_manager):
    mock_graph.flow_type = "DAG"
    mock_graph.nodes = {
        "node1": Node(
            definition=NodeDefinition(
                name="node1",
                type="Python",
                function="test_function",
                inputs={},
                outputs={},
            ),
            callable=Mock(),
        ),
        "node2": Node(
            definition=NodeDefinition(
                name="node2",
                type="Python",
                function="test_function",
                inputs={},
                outputs={},
            ),
            callable=Mock(),
        ),
    }
    mock_graph.dependencies = {"node1": [], "node2": []}

    async def mock_execute_node(node, input_data):
        await asyncio.sleep(0.1)
        return {"result": f"{node.definition.name}_output"}

    mock_node_manager.execute_node.side_effect = mock_execute_node

    result = await async_executor.execute({})

    assert isinstance(result, ExecutionContext)
    assert mock_node_manager.execute_node.call_count == 2
    assert result.context["node1"]["result"] == "node1_output"
    assert result.context["node2"]["result"] == "node2_output"


# Add more tests for error handling, condition evaluation, etc.
