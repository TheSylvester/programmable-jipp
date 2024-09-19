from abc import ABC, abstractmethod


class BasePromptTemplate(ABC):

    @property
    @abstractmethod
    def system_prompt(self):
        pass

    @property
    @abstractmethod
    def user_prompt(self):
        pass
