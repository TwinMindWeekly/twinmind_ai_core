import os
import time
import requests
import base64
import json
import numpy as np
import cv2
from typing import Union, Generator
from .base import BaseProvider

class GroqProvider(BaseProvider):
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("GROQ_API_KEY", "")
        self.vision_model = os.getenv("GROQ_VISION_MODEL", "llama-3.2-90b-vision-preview")
        self.text_model = os.getenv("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self._rate_limit_remaining = "?"
        self._rate_limit_reset = ""

    def _request_with_retry(self, payload, timeout=20, max_retries=2, stream=False):
        for attempt in range(max_retries):
            try:
                response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=timeout, stream=stream)
                self._parse_rate_headers(response)
                
                if response.status_code == 200:
                    self.request_count += 1
                    if stream:
                        return self._handle_stream(response)
                    else:
                        return response.json()["choices"][0]["message"]["content"].strip()
                elif response.status_code == 429 or "rate_limit" in response.text:
                    retry_after = response.headers.get("retry-after", "")
                    reset_req = response.headers.get("x-ratelimit-reset-requests", "")
                    wait = self._parse_wait_time(retry_after or reset_req, fallback=3 * (attempt + 1))
                    wait = min(wait, 10)
                    print(f"[Groq] Rate limit hit, waiting {wait:.0f}s... (retry {attempt + 1}/{max_retries})")
                    time.sleep(wait)
                    continue
                else:
                    print(f"[Groq] API error: {response.text}")
                    return "" if not stream else self._empty_stream()
            except Exception as e:
                print(f"[Groq] Connection error: {e}")
                return "" if not stream else self._empty_stream()
        print("[Groq] Max retries -> forward to fallback")
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
            print(f"[Groq] Stream error: {e}")
            yield ""

    def _empty_stream(self) -> Generator:
        yield ""

    @staticmethod
    def _parse_wait_time(header_value: str, fallback: float = 10.0) -> float:
        if not header_value: return fallback
        import re
        total = 0.0
        m = re.search(r'(\d+)m(?!\w*s)', header_value)
        if m: total += int(m.group(1)) * 60
        s = re.search(r'([\d.]+)s', header_value)
        if s: total += float(s.group(1))
        ms = re.search(r'(\d+)ms', header_value)
        if ms: total += int(ms.group(1)) / 1000
        return total + 1.0 if total > 0 else fallback

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
        return self._request_with_retry(payload, timeout=20 if not stream else 60, stream=stream)

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
        return self._request_with_retry(payload, timeout=30, stream=False)

    def _parse_rate_headers(self, response):
        h = response.headers
        self._rate_limit_remaining = h.get("x-ratelimit-remaining-requests", "?")
        self._rate_limit_reset = h.get("x-ratelimit-reset-requests", "")

    def get_usage_info(self) -> str:
        reset = f" | Reset: {self._rate_limit_reset}" if self._rate_limit_reset else ""
        return f"Groq | Used: {self.request_count} | Remaining: {self._rate_limit_remaining}{reset}"
