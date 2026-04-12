import os
import numpy as np
from typing import Union, Generator
from .providers.base import BaseProvider
from .providers.groq_provider import GroqProvider
from .providers.gemini_provider import GeminiProvider
from .providers.sambanova_provider import SambaNovaProvider
from .providers.ollama_provider import OllamaProvider

class AutoFallbackProvider(BaseProvider):
    def __init__(self):
        super().__init__()
        self._providers = []
        self._provider_names = []
        self._current_idx = 0

        if os.getenv("GROQ_API_KEY"):
            self._providers.append(GroqProvider())
            self._provider_names.append("Groq")
        if os.getenv("GEMINI_API_KEY"):
            self._providers.append(GeminiProvider())
            self._provider_names.append("Gemini")
        if os.getenv("SAMBANOVA_API_KEY"):
            self._providers.append(SambaNovaProvider())
            self._provider_names.append("SambaNova")

        if not self._providers:
            self._providers.append(OllamaProvider())
            self._provider_names.append("Ollama")

    def _current(self) -> BaseProvider:
        return self._providers[self._current_idx]

    def _current_name(self) -> str:
        return self._provider_names[self._current_idx]

    def _switch_next(self) -> bool:
        if self._current_idx + 1 < len(self._providers):
            self._current_idx += 1
            print(f"[AutoFallback] Switched to {self._current_name()}")
            return True
        return False

    def generate_text(self, system_prompt: str, user_prompt: str, stream: bool = False) -> Union[str, Generator]:
        start_idx = self._current_idx
        while True:
            try:
                result = self._current().generate_text(system_prompt, user_prompt, stream)
                
                # Check if generator is empty or if string is empty
                if stream:
                    generator = result
                    try:
                        first_item = next(generator)
                        if not first_item:
                            raise ValueError("Empty stream chunk")
                        self.request_count += 1
                        
                        def wrapped_generator():
                            yield first_item
                            yield from generator
                        
                        return wrapped_generator()
                    except (StopIteration, ValueError):
                        pass
                else:
                    if result:
                        self.request_count += 1
                        return result
            except Exception as e:
                print(f"[AutoFallback] Error with {self._current_name()}: {e}")

            if not self._switch_next():
                self._current_idx = start_idx
                return "" if not stream else self._empty_stream()

    def _empty_stream(self) -> Generator:
        yield ""

    def generate_vision(self, system_prompt: str, user_prompt: str, image_np: np.ndarray) -> str:
        start_idx = self._current_idx
        while True:
            result = self._current().generate_vision(system_prompt, user_prompt, image_np)
            if result:
                self.request_count += 1
                return result
            if not self._switch_next():
                self._current_idx = start_idx
                return ""

    def get_usage_info(self) -> str:
        parts = [f"{n}: {p.request_count}" for n, p in zip(self._provider_names, self._providers)]
        return f"AutoFallback [{self._current_name()}] | " + " | ".join(parts)
