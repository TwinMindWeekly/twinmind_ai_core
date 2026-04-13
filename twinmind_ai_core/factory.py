import os
from .providers.base import BaseProvider
from .providers.gemini_provider import GeminiProvider
from .providers.groq_provider import GroqProvider
from .providers.sambanova_provider import SambaNovaProvider
from .providers.ollama_provider import OllamaProvider
from .providers.local_provider import LocalProvider
from .fallback import AutoFallbackProvider

class AIProviderFactory:
    @staticmethod
    def get_provider(provider_name: str = None) -> BaseProvider:
        if not provider_name:
            provider_name = os.getenv("AI_PROVIDER", "Ollama").strip().lower()
        else:
            provider_name = provider_name.strip().lower()

        if provider_name == "auto":
            return AutoFallbackProvider()
        elif provider_name == "gemini":
            return GeminiProvider()
        elif provider_name == "groq":
            return GroqProvider()
        elif provider_name == "sambanova":
            return SambaNovaProvider()
        elif provider_name == "local":
            return LocalProvider()
        else:
            return OllamaProvider()
