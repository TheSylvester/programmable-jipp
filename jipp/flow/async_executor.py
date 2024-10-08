# async_executor.py

import asyncio
from typing import Dict, List, Any
import logging
from .node import Node
from .execution_context import ExecutionContext
from .node_manager import NodeManager
from .flow_utils import Graph
import collections

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
            execution_context.set_error(str(e))  # Add this line
        return execution_context

    async def _execute_dag(self, execution_context: ExecutionContext) -> None:
        """
        Executes a DAG flow considering node dependencies.

        Args:
            execution_context (ExecutionContext): The execution context.
        """
        dependencies = self.graph.dependencies
        execution_order = self._topological_sort(dependencies)
        for node_name in execution_order:
            node = self.graph.nodes[node_name]
            await self._execute_node(node, execution_context)

    def _topological_sort(self, dependencies: Dict[str, List[str]]) -> List[str]:
        in_degree = {u: 0 for u in dependencies}
        adjacency_list = {u: [] for u in dependencies}
        for u, deps in dependencies.items():
            for v in deps:
                if v in in_degree:
                    in_degree[u] += 1
                    adjacency_list[v].append(u)
                else:
                    raise ValueError(f"Undefined dependency: {v}")

        queue = collections.deque([u for u in dependencies if in_degree[u] == 0])
        sorted_order = []

        while queue:
            u = queue.popleft()
            sorted_order.append(u)
            for v in adjacency_list[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if len(sorted_order) != len(dependencies):
            raise ValueError("Cycle detected in node dependencies")

        return sorted_order

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
                        for node_name in then_nodes:
                            node = self.graph.nodes[node_name]
                            await self._execute_node(node, execution_context)
                elif "output" in step:
                    # Handle output mapping
                    outputs = {}
                    for key, value in step["output"].items():
                        try:
                            outputs[key] = execution_context.get_value(value)
                        except ValueError:
                            logger.warning(
                                f"Unable to resolve output value for key '{key}': {value}"
                            )
                    execution_context.set_output(outputs)
                else:
                    raise ValueError("Invalid step format in Flex flow")
            elif isinstance(step, list):
                for node_name in step:
                    node = self.graph.nodes[node_name]
                    await self._execute_node(node, execution_context)
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
        self, node: Node, execution_context: ExecutionContext
    ) -> None:
        """
        Executes a single node with retry logic.

        Args:
            node (Node): The node to execute.
            execution_context (ExecutionContext): The execution context.
        """
        max_retries = node.definition.max_retries
        retry_count = 0
        while retry_count <= max_retries:
            try:
                input_data = {
                    input_name: execution_context.get_value(input_value)
                    for input_name, input_value in node.definition.inputs.items()
                }
                start_time = asyncio.get_event_loop().time()
                logger.info(
                    f"Executing node '{node.definition.name}' of type '{node.definition.type}'"
                )
                outputs = await self.node_manager.execute_node(node, input_data)
                end_time = asyncio.get_event_loop().time()
                execution_time = end_time - start_time
                execution_context.set_node_output(node.definition.name, outputs)
                execution_context.metrics[node.definition.name] = {
                    "execution_time": execution_time
                }
                logger.info(
                    f"Node '{node.definition.name}' executed successfully in {execution_time:.2f}s"
                )

                # Call on_success callback if defined
                if node.definition.on_success:
                    node.definition.on_success(node, execution_context)

                return
            except Exception as e:
                logger.error(f"Node '{node.definition.name}' failed with error: {e}")
                retry_count += 1
                if retry_count > max_retries:
                    logger.error(
                        f"Node '{node.definition.name}' failed after {retry_count} attempts."
                    )
                    # Call on_failure callback if defined
                    if node.definition.on_failure:
                        node.definition.on_failure(e, node, execution_context)
                    raise e
                else:
                    logger.info(
                        f"Retrying node '{node.definition.name}' ({retry_count}/{max_retries})"
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
            pattern = r"\${([^}]+)}"
            matches = re.findall(pattern, condition)
            for match in matches:
                value = execution_context.get_value(f"${{{match}}}")
                condition = condition.replace(f"${{{match}}}", repr(value))
            # Evaluate the condition safely
            result = eval(condition)
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to evaluate condition '{condition}': {e}")
            return False
