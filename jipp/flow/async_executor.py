# async_executor.py

import asyncio
from typing import Dict, List, Any
import logging
from .node import BaseNode
from .execution_context import ExecutionContext
from .node_manager import NodeManager
from .flow_utils import Graph

logger = logging.getLogger(__name__)


class AsyncExecutor:
    """
    Executes the flow asynchronously, supporting both DAG and Flex flows.
    """

    def __init__(self, graph: Graph, node_manager: NodeManager):
        """
        Initializes the executor with a graph and node manager.

        Args:
            graph (Graph): The flow graph to execute.
            node_manager (NodeManager): The manager for node execution functions.
        """
        self.graph = graph
        self.node_manager = node_manager

    async def execute(self, input_data: Dict[str, Any]) -> ExecutionContext:
        """
        Executes the flow with the given input data.

        Args:
            input_data (Dict[str, Any]): The input data for the flow.

        Returns:
            ExecutionContext: The execution context after execution.
        """
        execution_context = ExecutionContext()
        execution_context.set_input(input_data)
        try:
            if self.graph.flow_type == "DAG":
                await self._execute_dag(execution_context)
            elif self.graph.flow_type == "Flex":
                await self._execute_flex(execution_context)
            else:
                raise ValueError(f"Unsupported flow type: {self.graph.flow_type}")
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            # Additional error handling can be added here
        return execution_context

    async def _execute_dag(self, execution_context: ExecutionContext) -> None:
        """
        Executes a DAG flow.

        Args:
            execution_context (ExecutionContext): The execution context.
        """
        for step in self.graph.flow_steps:
            if isinstance(step, list):
                await self._execute_nodes(step, execution_context)
            else:
                raise ValueError("Invalid step format in DAG flow")

    async def _execute_flex(self, execution_context: ExecutionContext) -> None:
        """
        Executes a Flex flow with conditional steps.

        Args:
            execution_context (ExecutionContext): The execution context.
        """
        for step in self.graph.flow_steps:
            if isinstance(step, dict):
                if "when" in step and "then" in step:
                    condition = step["when"]
                    then_nodes = step["then"]
                    if await self._evaluate_condition(condition, execution_context):
                        await self._execute_nodes(then_nodes, execution_context)
                elif "output" in step:
                    # Handle output mapping
                    outputs = {}
                    for key, value in step["output"].items():
                        outputs[key] = execution_context.get_value(value)
                    execution_context.set_output(outputs)
                else:
                    raise ValueError("Invalid step format in Flex flow")
            elif isinstance(step, list):
                await self._execute_nodes(step, execution_context)
            else:
                raise ValueError("Invalid step format in Flex flow")

    async def _execute_nodes(
        self, node_names: List[str], execution_context: ExecutionContext
    ) -> None:
        """
        Executes a list of nodes in parallel.

        Args:
            node_names (List[str]): The list of node names to execute.
            execution_context (ExecutionContext): The execution context.
        """
        tasks = []
        for node_name in node_names:
            node = self.graph.nodes[node_name]
            task = asyncio.create_task(self._execute_node(node, execution_context))
            tasks.append(task)
        await asyncio.gather(*tasks)

    async def _execute_node(
        self, node: BaseNode, execution_context: ExecutionContext
    ) -> None:
        """
        Executes a single node with retry logic.

        Args:
            node (BaseNode): The node to execute.
            execution_context (ExecutionContext): The execution context.
        """
        max_retries = node.max_retries
        on_failure = node.on_failure
        retry_count = 0
        while retry_count <= max_retries:
            try:
                input_data = {
                    input_name: execution_context.get_value(input_value)
                    for input_name, input_value in node.inputs.items()
                }
                start_time = asyncio.get_event_loop().time()
                logger.info(f"Executing node '{node.name}' of type '{node.type}'")
                outputs = await self.node_manager.execute_node(node, input_data)
                end_time = asyncio.get_event_loop().time()
                execution_time = end_time - start_time
                execution_context.set_node_output(node.name, outputs)
                execution_context.metrics[node.name] = {
                    "execution_time": execution_time
                }
                logger.info(
                    f"Node '{node.name}' executed successfully in {execution_time:.2f}s"
                )
                return
            except Exception as e:
                logger.error(f"Node '{node.name}' failed with error: {e}")
                retry_count += 1
                if retry_count > max_retries or on_failure != "retry":
                    logger.error(
                        f"Node '{node.name}' failed after {retry_count} attempts."
                    )
                    raise e
                else:
                    logger.info(
                        f"Retrying node '{node.name}' ({retry_count}/{max_retries})"
                    )
                    await asyncio.sleep(1)

    async def _evaluate_condition(
        self, condition: str, execution_context: ExecutionContext
    ) -> bool:
        """
        Evaluates a condition for conditional execution.

        Args:
            condition (str): The condition to evaluate.
            execution_context (ExecutionContext): The execution context.

        Returns:
            bool: The result of the condition.
        """
        try:
            import re

            # Replace references in condition with actual values
            pattern = r"\${[^}]+}"
            matches = re.findall(pattern, condition)
            for match in matches:
                value = execution_context.get_value(match)
                condition = condition.replace(match, repr(value))
            # Evaluate the condition safely
            result = eval(condition)
            return result
        except Exception as e:
            logger.error(f"Failed to evaluate condition '{condition}': {e}")
            return False
