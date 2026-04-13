import os
import cv2
import numpy as np
from PIL import Image
from typing import Union, Generator
from .base import BaseProvider

class GeminiProvider(BaseProvider):
    def __init__(self):
        super().__init__()
        from google import genai
        from google.genai import types
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    def _empty_stream(self) -> Generator:
        yield ""

    def generate_text(self, system_prompt: str, user_prompt: str, stream: bool = False) -> Union[str, Generator]:
        if not self.client:
            print("[Gemini] Error: API Key is missing")
            return "" if not stream else self._empty_stream()

        prompt = ""
        if system_prompt:
            prompt += f"{system_prompt}\n\n"
        if user_prompt:
            prompt += user_prompt

        try:
            from google.genai import types
            safety_settings = [
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE)
            ]
            generation_config = types.GenerateContentConfig(temperature=0.1, top_p=0.5, safety_settings=safety_settings)
            if stream:
                response = self.client.models.generate_content_stream(model=self.model_name, contents=prompt, config=generation_config)
                self.request_count += 1
                return self._handle_stream(response)
            else:
                response = self.client.models.generate_content(model=self.model_name, contents=prompt, config=generation_config)
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
        if image_np is None or image_np.size == 0 or not self.client:
            return ""
            
        prompt = ""
        if system_prompt:
            prompt += f"{system_prompt}\n\n"
        if user_prompt:
            prompt += user_prompt

        try:
            img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            from google.genai import types
            safety_settings = [
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE)
            ]
            generation_config = types.GenerateContentConfig(temperature=0.1, top_p=0.5, safety_settings=safety_settings)
            response = self.client.models.generate_content(model=self.model_name, contents=[prompt, pil_img], config=generation_config)
            self.request_count += 1
            return response.text.strip()
        except Exception as e:
            print(f"[Gemini Vision] Error: {e}")
            return ""

    def get_usage_info(self) -> str:
        remaining = max(0, 1500 - self.request_count)
        return f"Gemini ({self.model_name}) | Used: {self.request_count} | ~{remaining} left/day"
