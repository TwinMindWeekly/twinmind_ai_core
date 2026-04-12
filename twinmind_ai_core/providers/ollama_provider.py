import os
import requests
import base64
import json
import cv2
import numpy as np
from typing import Union, Generator
from .base import BaseProvider

class OllamaProvider(BaseProvider):
    def __init__(self):
        super().__init__()
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "gemma:2b")
        
    def _empty_stream(self) -> Generator:
        yield ""

    def generate_text(self, system_prompt: str, user_prompt: str, stream: bool = False) -> Union[str, Generator]:
        prompt = ""
        if system_prompt:
            prompt += f"{system_prompt}\n\n"
        if user_prompt:
            prompt += user_prompt

        if not prompt:
            return "" if not stream else self._empty_stream()

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": 0.1,
                "top_p": 0.5
            }
        }

        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=20 if not stream else 60, stream=stream)
            response.raise_for_status()
            self.request_count += 1
            if stream:
                return self._handle_stream(response)
            data = response.json()
            return data.get("response", "").strip()
        except Exception as e:
            print(f"[Ollama] Connection error: {e}")
            return "" if not stream else self._empty_stream()
            
    def _handle_stream(self, response) -> Generator:
        try:
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    data = json.loads(line)
                    chunk = data.get("response", "")
                    if chunk:
                        yield chunk
        except Exception as e:
            print(f"[Ollama] Stream error: {e}")
            yield ""

    def generate_vision(self, system_prompt: str, user_prompt: str, image_np: np.ndarray) -> str:
        if image_np is None or image_np.size == 0:
            return ""

        prompt = ""
        if system_prompt:
            prompt += f"{system_prompt}\n\n"
        if user_prompt:
            prompt += user_prompt

        _, buffer = cv2.imencode('.jpg', image_np)
        b64_str = base64.b64encode(buffer).decode('utf-8')
        
        vision_model = os.getenv("OLLAMA_VISION_MODEL", "llama3.2-vision")
        
        payload = {
            "model": vision_model,
            "prompt": prompt,
            "stream": False,
            "images": [b64_str],
            "options": {
                "temperature": 0.1,
                "top_p": 0.5
            }
        }
        
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
            if response.status_code == 200:
                self.request_count += 1
                result = response.json()
                return result.get("response", "").strip()
            else:
                print(f"[Ollama Vision] API Error: {response.text}")
                return ""
        except Exception as e:
            print(f"[Ollama Vision] Connection error: {e}")
            return ""

    def get_usage_info(self) -> str:
        return f"Ollama (Local) | Requests: {self.request_count} | No limits"
