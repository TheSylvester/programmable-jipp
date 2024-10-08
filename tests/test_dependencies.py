import pytest
from jipp.flow.flow_utils import Graph, adapt_node
from jipp.flow.async_executor import AsyncExecutor
from jipp.flow.node_manager import NodeManager
from jipp.flow.node import Node, NodeDefinition
from jipp.flow.execution_context import ExecutionContext
from unittest.mock import AsyncMock, Mock


@pytest.fixture
def node_manager():
    mock_manager = NodeManager()
    mock_manager.execute_node = AsyncMock(return_value={"result": "success"})
    return mock_manager


def on_success_callback(node, context):
    print(f"Node {node.definition.name} succeeded")


def on_failure_callback(exception, node, context):
    print(f"Node {node.definition.name} failed with error: {str(exception)}")


def test_dependency_parsing_linear():
    flow_steps = [
        {"name": "node_a", "type": "Python", "dependencies": []},
        {"name": "node_b", "type": "LLM", "dependencies": ["node_a"]},
        {"name": "node_c", "type": "API", "dependencies": ["node_b"]},
    ]
    graph = Graph(nodes={}, flow_steps=flow_steps, flow_type="DAG")
    assert graph.dependencies == {
        "node_a": [],
        "node_b": ["node_a"],
        "node_c": ["node_b"],
    }


def test_dependency_parsing_branching():
    flow_steps = [
        {"name": "node_a", "type": "Python", "dependencies": []},
        {"name": "node_b", "type": "LLM", "dependencies": ["node_a"]},
        {"name": "node_c", "type": "API", "dependencies": ["node_a"]},
        {"name": "node_d", "type": "Python", "dependencies": ["node_b", "node_c"]},
    ]
    graph = Graph(nodes={}, flow_steps=flow_steps, flow_type="DAG")
    assert graph.dependencies == {
        "node_a": [],
        "node_b": ["node_a"],
        "node_c": ["node_a"],
        "node_d": ["node_b", "node_c"],
    }


def test_dependency_parsing_cyclic():
    flow_steps = [
        {"name": "node_a", "type": "Python", "dependencies": ["node_c"]},
        {"name": "node_b", "type": "LLM", "dependencies": ["node_a"]},
        {"name": "node_c", "type": "API", "dependencies": ["node_b"]},
    ]
    graph = Graph(nodes={}, flow_steps=flow_steps, flow_type="DAG")
    executor = AsyncExecutor(graph, NodeManager())
    with pytest.raises(ValueError, match="Cycle detected in node dependencies"):
        executor._topological_sort(graph.dependencies)


@pytest.mark.asyncio
async def test_execute_dag_linear(node_manager):
    flow_steps = [
        {"name": "node_a", "type": "Python", "dependencies": []},
        {"name": "node_b", "type": "LLM", "dependencies": ["node_a"]},
        {"name": "node_c", "type": "API", "dependencies": ["node_b"]},
    ]
    nodes = {
        "node_a": Node(
            definition=NodeDefinition(
                name="node_a",
                type="Python",
                function="func_a",
                inputs={},
                outputs={},
            ),
            callable=lambda: None,
        ),
        "node_b": Node(
            definition=NodeDefinition(
                name="node_b",
                type="LLM",
                function="func_b",
                inputs={},
                outputs={},
            ),
            callable=lambda: None,
        ),
        "node_c": Node(
            definition=NodeDefinition(
                name="node_c",
                type="API",
                function="func_c",
                inputs={},
                outputs={},
            ),
            callable=lambda: None,
        ),
    }
    graph = Graph(nodes=nodes, flow_steps=flow_steps, flow_type="DAG")
    executor = AsyncExecutor(graph, node_manager)
    execution_context = await executor.execute({})

    # Verify execute_node was called in order
    assert node_manager.execute_node.call_args_list[0][0][0].definition.name == "node_a"
    assert node_manager.execute_node.call_args_list[1][0][0].definition.name == "node_b"
    assert node_manager.execute_node.call_args_list[2][0][0].definition.name == "node_c"


@pytest.mark.asyncio
async def test_execute_dag_branching(node_manager):
    flow_steps = [
        {"name": "node_a", "type": "Python", "dependencies": []},
        {"name": "node_b", "type": "LLM", "dependencies": ["node_a"]},
        {"name": "node_c", "type": "API", "dependencies": ["node_a"]},
        {"name": "node_d", "type": "Python", "dependencies": ["node_b", "node_c"]},
    ]
    nodes = {
        "node_a": Node(
            definition=NodeDefinition(
                name="node_a",
                type="Python",
                function="func_a",
                inputs={},
                outputs={},
                max_retries=0,
                on_success=on_success_callback,
                on_failure=on_failure_callback,
            ),
            callable=lambda: None,
        ),
        "node_b": Node(
            definition=NodeDefinition(
                name="node_b",
                type="LLM",
                function="func_b",
                inputs={},
                outputs={},
                max_retries=0,
                on_success=on_success_callback,
                on_failure=on_failure_callback,
            ),
            callable=lambda: None,
        ),
        "node_c": Node(
            definition=NodeDefinition(
                name="node_c",
                type="API",
                function="func_c",
                inputs={},
                outputs={},
                max_retries=0,
                on_success=on_success_callback,
                on_failure=on_failure_callback,
            ),
            callable=lambda: None,
        ),
        "node_d": Node(
            definition=NodeDefinition(
                name="node_d",
                type="Python",
                function="func_d",
                inputs={},
                outputs={},
                max_retries=0,
                on_success=on_success_callback,
                on_failure=on_failure_callback,
            ),
            callable=lambda: None,
        ),
    }
    graph = Graph(nodes=nodes, flow_steps=flow_steps, flow_type="DAG")
    executor = AsyncExecutor(graph, node_manager)
    execution_context = await executor.execute({})

    # Verify execute_node was called in correct order with branching
    assert node_manager.execute_node.call_args_list[0][0][0].definition.name == "node_a"
    assert node_manager.execute_node.call_args_list[1][0][0].definition.name in [
        "node_b",
        "node_c",
    ]
    assert node_manager.execute_node.call_args_list[2][0][0].definition.name in [
        "node_b",
        "node_c",
    ]
    assert node_manager.execute_node.call_args_list[3][0][0].definition.name == "node_d"


# Additional tests can be added as needed
