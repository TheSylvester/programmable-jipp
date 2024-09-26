import asyncio
import importlib
import yaml
import uuid
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, create_model
from jipp.utils.logging_utils import log
from jipp.jipp_engine import ask_llm
import httpx


class NodeDefinition(BaseModel):
    name: str
    type: str
    function: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]


class Node(BaseModel):
    definition: NodeDefinition
    callable: Callable


class NodeError(Exception):
    pass


class FlowExecutionError(Exception):
    pass


class NodeLoader:
    @staticmethod
    def load(node_def: NodeDefinition) -> Node:
        if node_def.type == "llm":
            return Node(definition=node_def, callable=ask_llm)
        elif node_def.type == "api":
            return Node(definition=node_def, callable=NodeLoader.api_call)
        else:
            try:
                module_name, function_name = node_def.function.rsplit(".", 1)
                module = importlib.import_module(module_name)
                function = getattr(module, function_name)
                return Node(definition=node_def, callable=function)
            except (ImportError, AttributeError) as e:
                raise NodeError(f"Failed to load node {node_def.name}: {str(e)}")

    @staticmethod
    async def api_call(**kwargs):
        url = kwargs.pop("url")
        method = kwargs.pop("method", "GET")
        async with httpx.AsyncClient() as client:
            response = await client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()


class NodeManager:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}

    def load_nodes(self, node_definitions: List[Dict[str, Any]]):
        for node_def in node_definitions:
            self.load_node(NodeDefinition(**node_def))

    def load_node(self, node_def: NodeDefinition):
        node = NodeLoader.load(node_def)
        self.nodes[node_def.name] = node

    def get_node(self, node_name: str) -> Optional[Node]:
        return self.nodes.get(node_name)

    def validate_nodes(self):
        for node_name, node in self.nodes.items():
            self._validate_node_inputs(node_name, node)
            self._validate_node_outputs(node_name, node)

    def _validate_node_inputs(self, node_name: str, node: Node):
        expected_inputs = set(node.definition.inputs.keys())
        actual_inputs = set(node.callable.__annotations__.keys()) - {"return"}
        if expected_inputs != actual_inputs:
            raise NodeError(
                f"Input mismatch for node {node_name}. "
                f"Expected: {expected_inputs}, Actual: {actual_inputs}"
            )

    def _validate_node_outputs(self, node_name: str, node: Node):
        expected_outputs = set(node.definition.outputs.keys())
        if "return" in node.callable.__annotations__:
            return_annotation = node.callable.__annotations__["return"]
            if isinstance(return_annotation, type) and issubclass(
                return_annotation, BaseModel
            ):
                actual_outputs = set(return_annotation.__fields__.keys())
                if expected_outputs != actual_outputs:
                    raise NodeError(
                        f"Output mismatch for node {node_name}. "
                        f"Expected: {expected_outputs}, Actual: {actual_outputs}"
                    )
            else:
                log.warning(
                    f"Unable to validate outputs for node {node_name}. Return type is not a Pydantic model."
                )
        else:
            log.warning(
                f"Unable to validate outputs for node {node_name}. No return annotation found."
            )


class FlowExecutionContext:
    def __init__(self, flow_id: str, run_id: str):
        self.flow_id = flow_id
        self.run_id = run_id
        self.node_runs = {}


class FlowExecutor:
    def __init__(self, flow_file: str):
        self.flow_file = flow_file
        self.flow = self.load_flow()
        self.node_manager = NodeManager()
        self.node_manager.load_nodes(self.flow["nodes"])
        self.node_manager.validate_nodes()
        self.node_outputs = {}
        self.context = FlowExecutionContext(
            self.flow.get("id", "default_flow"), str(uuid.uuid4())
        )

    def load_flow(self) -> Dict[str, Any]:
        with open(self.flow_file, "r") as file:
            return yaml.safe_load(file)

    async def execute_node(self, node_def: Dict[str, Any]) -> Any:
        node_name = node_def["name"]
        log.info(f"Executing node: {node_name}")

        node = self.node_manager.get_node(node_name)
        if not node:
            raise FlowExecutionError(f"Node {node_name} not found")

        try:
            evaluated_inputs = await self.evaluate_inputs(node.definition.inputs)

            if node.definition.type == "llm":
                result = await self.execute_llm_node(node.callable, evaluated_inputs)
            elif asyncio.iscoroutinefunction(node.callable):
                result = await node.callable(**evaluated_inputs)
            else:
                result = await asyncio.to_thread(node.callable, **evaluated_inputs)

            self.store_outputs(node_name, node.definition.outputs, result)
            log.info(f"Node {node_name} executed successfully")
            return result

        except Exception as e:
            log.error(f"Error executing node {node_name}: {str(e)}")
            raise FlowExecutionError(f"Error executing node {node_name}: {str(e)}")

    async def execute_llm_node(self, function: Callable, inputs: Dict[str, Any]) -> Any:
        response_format = inputs.get("response_format")
        if response_format:
            dynamic_model = create_model("DynamicModel", **response_format)
            inputs["response_format"] = dynamic_model
        return await function(**inputs)

    async def evaluate_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        evaluated = {}
        for key, value in inputs.items():
            if (
                isinstance(value, str)
                and value.startswith("${")
                and value.endswith("}")
            ):
                node_name, output_name = value[2:-1].split(".", 1)
                evaluated[key] = await self.get_node_output(node_name, output_name)
            else:
                evaluated[key] = value
        return evaluated

    async def get_node_output(self, node_name: str, output_name: str) -> Any:
        if f"{node_name}.{output_name}" not in self.node_outputs:
            node = next((n for n in self.flow["nodes"] if n["name"] == node_name), None)
            if node:
                await self.execute_node(node)
            else:
                raise FlowExecutionError(f"Node {node_name} not found in flow")
        return self.node_outputs.get(f"{node_name}.{output_name}")

    def store_outputs(self, node_name: str, outputs: Dict[str, Any], result: Any):
        if isinstance(result, BaseModel):
            result = result.model_dump()
        if isinstance(result, dict):
            for output_name in outputs:
                self.node_outputs[f"{node_name}.{output_name}"] = result.get(
                    output_name
                )
        else:
            for output_name in outputs:
                self.node_outputs[f"{node_name}.{output_name}"] = result

    async def execute(self):
        for node in self.flow["nodes"]:
            await self.execute_node(node)

        flow_outputs = self.flow.get("outputs", {})
        return {
            key: await self.get_node_output(*value.split("."))
            for key, value in flow_outputs.items()
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python flow_executor.py <flow_file>")
        sys.exit(1)

    flow_file = sys.argv[1]
    executor = FlowExecutor(flow_file)
    result = asyncio.run(executor.execute())
    print("Flow execution result:", result)
