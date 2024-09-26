# node.py

import importlib
from pydantic import BaseModel, Field, model_validator
from typing import Any, Dict, Optional, Union, Callable


class BaseNode(BaseModel):
    """
    Base class for all node types.
    """

    name: str
    type: str  # 'Python', 'LLM', 'API', etc.
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    max_retries: int = 0
    on_failure: Optional[str] = None
    function: str  # Path to the function (e.g., "module.submodule.function_name")

    @model_validator(mode="before")
    @classmethod
    def check_required_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        node_type = values.get("type")
        if not node_type:
            raise ValueError("Node 'type' is required.")
        return values

    def get_callable(self) -> Callable:
        module_name, function_name = self.function.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, function_name)


class PythonNode(BaseNode):
    """
    Node representing a Python script execution.
    """

    type: str = "Python"


class LLMNode(BaseNode):
    """
    Node representing a Large Language Model (LLM) execution.
    """

    type: str = "LLM"
    connection: str  # e.g., 'openai'
    prompt_template: str


class APINode(BaseNode):
    """
    Node representing an API call.
    """

    type: str = "API"
    api: str  # API endpoint or identifier


# Union of all node types for type checking
Node = Union[PythonNode, LLMNode, APINode]
