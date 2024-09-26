# main.py

import asyncio
import logging
import os
from typing import Any, Dict
from jipp.flow.node_manager import NodeManager
from jipp.flow.async_executor import AsyncExecutor
from jipp.flow.flow_utils import load_flow_from_yaml
from jipp.flow.execution_context import ExecutionContext
from jipp.flow.node import BaseNode, PythonNode, LLMNode, APINode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Node execution functions
async def execute_python(
    node: PythonNode, input_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Executes a Python node.

    Args:
        node (PythonNode): The Python node to execute.
        input_data (Dict[str, Any]): The input data for the node.

    Returns:
        Dict[str, Any]: The outputs of the node.
    """
    source = node.source
    # Execute the Python code
    exec_globals = {}
    exec_locals = input_data.copy()
    exec(source, exec_globals, exec_locals)
    # Collect outputs
    outputs = {key: exec_locals.get(key) for key in node.outputs.keys()}
    return outputs


async def execute_llm(node: LLMNode, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes an LLM node.

    Args:
        node (LLMNode): The LLM node to execute.
        input_data (Dict[str, Any]): The input data for the node.

    Returns:
        Dict[str, Any]: The outputs of the node.
    """
    prompt_template = node.prompt_template
    connection = node.connection
    # Retrieve API key from environment variable
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError("LLM API key not found in environment variables.")
    # Render the prompt with input data
    prompt = prompt_template.format(**input_data)
    # Placeholder for LLM API call
    await asyncio.sleep(0.1)
    return {"output": f"LLM response to: {prompt}"}


async def execute_api(node: APINode, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes an API node.

    Args:
        node (APINode): The API node to execute.
        input_data (Dict[str, Any]): The input data for the node.

    Returns:
        Dict[str, Any]: The outputs of the node.
    """
    api_endpoint = node.api
    # Retrieve API token from environment variable
    api_token = os.getenv("API_TOKEN")
    if not api_token:
        raise ValueError("API token not found in environment variables.")
    # Placeholder for API call
    await asyncio.sleep(0.1)
    return {"output": f"API response from {api_endpoint}"}


# Register node handlers
node_manager = NodeManager()
node_manager.register_node_type("Python", execute_python)
node_manager.register_node_type("LLM", execute_llm)
node_manager.register_node_type("API", execute_api)


# Main execution
if __name__ == "__main__":
    # Load the graph from a YAML file
    graph = load_flow_from_yaml("dag_flow.yaml")  # For DAG flow
    # graph = load_flow_from_yaml('flex_flow.yaml')  # For Flex flow

    # Create an AsyncExecutor
    executor = AsyncExecutor(graph, node_manager)

    # Prepare input data
    input_data = {
        "url": "https://example.com",
        "urls": ["https://example.com", "https://example.org"],
    }

    # Run the executor
    async def main() -> None:
        execution_context = await executor.execute(input_data)
        # Retrieve outputs
        context_data = execution_context.get_context()
        print("Execution Results:")
        for node_name, outputs in context_data.items():
            print(f"{node_name}: {outputs}")
        # Print flow-level outputs if any
        print("\nFlow Outputs:")
        print(execution_context.outputs)
        # Print metrics
        print("\nExecution Metrics:")
        for node_name, metrics in execution_context.metrics.items():
            print(f"{node_name}: {metrics}")

    asyncio.run(main())
