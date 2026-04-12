import numpy as np
from typing import Union, Generator
from abc import ABC, abstractmethod

class BaseProvider(ABC):
    def __init__(self):
        self.request_count = 0

    @abstractmethod
    def generate_text(self, system_prompt: str, user_prompt: str, stream: bool = False) -> Union[str, Generator]:
        pass

    @abstractmethod
    def generate_vision(self, system_prompt: str, user_prompt: str, image_np: np.ndarray) -> str:
        pass

    @abstractmethod
    def get_usage_info(self) -> str:
        return f"Requests: {self.request_count}"
