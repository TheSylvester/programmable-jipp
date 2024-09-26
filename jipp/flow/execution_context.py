# execution_context.py

from pydantic import BaseModel, Field
from typing import Any, Dict


class ExecutionContext(BaseModel):
    """
    Manages the execution context, including node outputs and metrics.
    """

    context: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    cache: Dict[str, Any] = Field(default_factory=dict)  # Simple caching mechanism

    def set_node_output(self, node_name: str, outputs: Dict[str, Any]) -> None:
        """
        Stores the outputs of a node.

        Args:
            node_name (str): The name of the node.
            outputs (Dict[str, Any]): The outputs to store.
        """
        self.context[node_name] = outputs
        self.cache[node_name] = outputs  # Cache the outputs

    def get_value(self, reference: Any) -> Any:
        """
        Resolves a reference to a value in the context.

        Args:
            reference (Any): The reference to resolve.

        Returns:
            Any: The resolved value.

        Raises:
            ValueError: If the reference cannot be resolved.
        """
        if (
            isinstance(reference, str)
            and reference.startswith("${")
            and reference.endswith("}")
        ):
            ref = reference[2:-1]  # Strip '${' and '}'
            parts = ref.split(".")
            value = self.context
            for part in parts:
                if part == "input":
                    value = self.context.get("input", {})
                else:
                    value = value.get(part)
                if value is None:
                    raise ValueError(
                        f"Reference '{reference}' not found in execution context."
                    )
            return value
        else:
            return reference

    def set_input(self, input_data: Dict[str, Any]) -> None:
        """
        Sets the flow-level input data.

        Args:
            input_data (Dict[str, Any]): The input data.
        """
        self.context["input"] = input_data

    def set_output(self, outputs: Dict[str, Any]) -> None:
        """
        Sets the flow-level outputs.

        Args:
            outputs (Dict[str, Any]): The outputs to set.
        """
        self.outputs.update(outputs)

    def get_context(self) -> Dict[str, Any]:
        """
        Retrieves the entire execution context.

        Returns:
            Dict[str, Any]: The execution context.
        """
        return self.context
