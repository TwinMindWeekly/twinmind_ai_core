import os
import cv2
import numpy as np
from PIL import Image
from typing import Union, Generator
from .base import BaseProvider

class GeminiProvider(BaseProvider):
    def __init__(self):
        super().__init__()
        import google.generativeai as genai
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        else:
            self.model = None

    def _empty_stream(self) -> Generator:
        yield ""

    def generate_text(self, system_prompt: str, user_prompt: str, stream: bool = False) -> Union[str, Generator]:
        if not self.model:
            print("[Gemini] Error: API Key is missing")
            return "" if not stream else self._empty_stream()

        prompt = ""
        if system_prompt:
            prompt += f"{system_prompt}\n\n"
        if user_prompt:
            prompt += user_prompt

        try:
            generation_config = {"temperature": 0.1, "top_p": 0.5}
            if stream:
                response = self.model.generate_content(prompt, generation_config=generation_config, stream=True)
                self.request_count += 1
                return self._handle_stream(response)
            else:
                response = self.model.generate_content(prompt, generation_config=generation_config)
                self.request_count += 1
                return response.text.strip()
        except Exception as e:
            print(f"[Gemini] Error: {e}")
            return "" if not stream else self._empty_stream()

    def _handle_stream(self, response) -> Generator:
        try:
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            print(f"[Gemini] Stream error: {e}")
            yield ""

    def generate_vision(self, system_prompt: str, user_prompt: str, image_np: np.ndarray) -> str:
        if image_np is None or image_np.size == 0 or not self.model:
            return ""
            
        prompt = ""
        if system_prompt:
            prompt += f"{system_prompt}\n\n"
        if user_prompt:
            prompt += user_prompt

        try:
            img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            generation_config = {"temperature": 0.1, "top_p": 0.5}
            response = self.model.generate_content([prompt, pil_img], generation_config=generation_config)
            self.request_count += 1
            return response.text.strip()
        except Exception as e:
            print(f"[Gemini Vision] Error: {e}")
            return ""

    def get_usage_info(self) -> str:
        remaining = max(0, 1500 - self.request_count)
        return f"Gemini ({self.model_name}) | Used: {self.request_count} | ~{remaining} left/day"
