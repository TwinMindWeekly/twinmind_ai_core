import os
import requests
import base64
import cv2
import json
import numpy as np
from typing import Union, Generator
from .base import BaseProvider

class SambaNovaProvider(BaseProvider):
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("SAMBANOVA_API_KEY", "")
        self.text_model = os.getenv("SAMBANOVA_TEXT_MODEL", "Meta-Llama-3.3-70B-Instruct")
        self.vision_model = os.getenv("SAMBANOVA_VISION_MODEL", "Llama-4-Maverick-17B-128E-Instruct")
        self.base_url = "https://api.sambanova.ai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _empty_stream(self) -> Generator:
        yield ""

    def generate_text(self, system_prompt: str, user_prompt: str, stream: bool = False) -> Union[str, Generator]:
        if not self.api_key:
            return "" if not stream else self._empty_stream()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.text_model,
            "messages": messages,
            "temperature": 0.1,
            "stream": stream
        }
        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=30, stream=stream)
            if response.status_code == 200:
                self.request_count += 1
                if stream:
                    return self._handle_stream(response)
                return response.json()["choices"][0]["message"]["content"].strip()
            elif response.status_code == 429:
                print("[SambaNova] Rate limit hit")
                return "" if not stream else self._empty_stream()
            else:
                print(f"[SambaNova] API error: {response.text}")
                return "" if not stream else self._empty_stream()
        except Exception as e:
            print(f"[SambaNova] Connection error: {e}")
            return "" if not stream else self._empty_stream()
            
    def _handle_stream(self, response) -> Generator:
        try:
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: ') and line != 'data: [DONE]':
                        data = json.loads(line[6:])
                        chunk = data["choices"][0].get("delta", {}).get("content", "")
                        if chunk:
                            yield chunk
        except Exception as e:
            print(f"[SambaNova] Stream error: {e}")
            yield ""

    def generate_vision(self, system_prompt: str, user_prompt: str, image_np: np.ndarray) -> str:
        if image_np is None or image_np.size == 0 or not self.api_key:
            return ""

        _, buffer = cv2.imencode('.jpg', image_np)
        b64_str = base64.b64encode(buffer).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{b64_str}"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content = []
        if user_prompt:
            user_content.append({"type": "text", "text": user_prompt})
        user_content.append({"type": "image_url", "image_url": {"url": image_url}})

        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": self.vision_model,
            "messages": messages,
            "temperature": 0.1,
            "stream": False
        }
        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=30)
            if response.status_code == 200:
                self.request_count += 1
                return response.json()["choices"][0]["message"]["content"].strip()
            elif response.status_code == 429:
                print("[SambaNova] Rate limit hit (vision)")
                return ""
            else:
                print(f"[SambaNova Vision] API error: {response.text}")
                return ""
        except Exception as e:
            print(f"[SambaNova Vision] Connection error: {e}")
            return ""

    def get_usage_info(self) -> str:
        remaining = max(0, 20 - self.request_count)
        return f"SambaNova ({self.text_model}) | Used: {self.request_count} | ~{remaining} left/day"
