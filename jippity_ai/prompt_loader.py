import os
from typing import List


class PromptManager:
    def __init__(self):
        self._prompts_dir = None

    def load_prompts(self, directory: str) -> None:
        self._prompts_dir = directory

    def __getattr__(self, name: str) -> str:
        if self._prompts_dir is None:
            raise RuntimeError("Prompts directory not set. Call load_prompts first.")

        file_path = os.path.join(self._prompts_dir, f"{name}.md")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read().strip()
        raise AttributeError(f"No prompt named '{name}' found")

    def list_prompts(self) -> List[str]:
        if self._prompts_dir is None:
            raise RuntimeError("Prompts directory not set. Call load_prompts first.")
        return [
            os.path.splitext(f)[0]
            for f in os.listdir(self._prompts_dir)
            if f.endswith(".md")
        ]


PROMPTS = PromptManager()
