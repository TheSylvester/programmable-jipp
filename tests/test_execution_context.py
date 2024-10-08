import pytest
from jipp.flow.execution_context import ExecutionContext


@pytest.fixture
def execution_context():
    return ExecutionContext()


def test_set_and_get_input(execution_context):
    execution_context.set_input({"input": "test_value"})
    assert execution_context.get_value("${input}") == "test_value"


def test_set_and_get_node_output(execution_context):
    execution_context.set_node_output("node1", {"result": "node1_output"})
    assert execution_context.get_value("${node1.result}") == "node1_output"


def test_nested_reference(execution_context):
    execution_context.set_node_output("node1", {"nested": {"value": "nested_output"}})
    assert execution_context.get_value("${node1.nested.value}") == "nested_output"


def test_non_existent_reference(execution_context):
    with pytest.raises(
        ValueError,
        match="Reference '\\$\\{non_existent\\}' not found in execution context.",
    ):
        execution_context.get_value("${non_existent}")


def test_caching_mechanism(execution_context):
    execution_context.set_node_output("node1", {"result": "original_output"})
    assert execution_context.get_value("${node1.result}") == "original_output"

    # Simulate a cache update
    execution_context.cache["node1.result"] = "cached_output"
    assert execution_context.get_value("${node1.result}") == "cached_output"


def test_set_and_get_output(execution_context):
    execution_context.set_output({"final_result": "flow_output"})
    assert execution_context.outputs == {"final_result": "flow_output"}


def test_get_context(execution_context):
    execution_context.set_input({"input": "test_input"})
    execution_context.set_node_output("node1", {"result": "node1_output"})
    expected_context = {"input": "test_input", "node1": {"result": "node1_output"}}
    assert execution_context.get_context() == expected_context


def test_set_and_get_error(execution_context):
    execution_context.set_error("Test error message")
    assert execution_context.error == "Test error message"


def test_reference_with_non_string_value(execution_context):
    execution_context.set_node_output(
        "node1", {"number": 42, "boolean": True, "list": [1, 2, 3]}
    )
    assert execution_context.get_value("${node1.number}") == 42
    assert execution_context.get_value("${node1.boolean}") is True
    assert execution_context.get_value("${node1.list}") == [1, 2, 3]


def test_get_value_with_non_reference(execution_context):
    assert execution_context.get_value("plain_string") == "plain_string"
    assert execution_context.get_value(42) == 42
    assert execution_context.get_value(True) is True


def test_metrics(execution_context):
    execution_context.metrics["node1"] = {"execution_time": 0.5}
    assert execution_context.metrics["node1"]["execution_time"] == 0.5


def test_complex_nested_reference(execution_context):
    execution_context.set_node_output(
        "node1", {"level1": {"level2": {"level3": "deep_nested_value"}}}
    )
    assert (
        execution_context.get_value("${node1.level1.level2.level3}")
        == "deep_nested_value"
    )


def test_reference_in_input(execution_context):
    execution_context.set_input({"input1": "value1", "input2": "${input1}"})
    assert (
        execution_context.get_value("${input2}") == "${input1}"
    )  # It should not resolve nested references in input
    assert execution_context.get_value("${input1}") == "value1"


def test_set_input_overwrite(execution_context):
    execution_context.set_input({"input": "initial_value"})
    execution_context.set_input({"input": "new_value"})
    assert execution_context.get_value("${input}") == "new_value"


def test_set_output_overwrite(execution_context):
    execution_context.set_output({"output": "initial_output"})
    execution_context.set_output({"output": "new_output"})
    assert execution_context.outputs["output"] == "new_output"
