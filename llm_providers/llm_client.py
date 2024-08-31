from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def generate_response(self, **kwargs):
        """Generate a response from the LLM using any parameters needed."""
        pass
