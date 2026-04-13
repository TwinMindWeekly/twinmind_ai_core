"""Local offline translation provider using MarianMT (Helsinki-NLP/opus-mt-*).

This provider runs translation fully offline on CPU or CUDA, similar to the
video_translator bridge. Models are lazy-loaded per (source, target) language
pair and cached. Vision input is NOT supported — ``generate_vision`` returns
an empty string so callers can fall back to OCR + text translation.

The ``translate_direct(text, src_lang, tgt_lang)`` method is the preferred API:
callers that already have raw text and language codes should use it to bypass
LLM-style prompt parsing. The ``BaseProvider`` methods (``generate_text`` /
``generate_vision``) are kept for interface compatibility.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, Generator, Optional, Tuple, Union

import numpy as np

from .base import BaseProvider

_logger = logging.getLogger(__name__)

# Direct MarianMT pairs known to exist on Hugging Face (non-exhaustive).
# When a direct pair is missing we pivot through English.
_KNOWN_DIRECT_PAIRS = {
    ("en", "vi"),
    ("vi", "en"),
    ("zh", "vi"),
    ("en", "ja"),
    ("ja", "en"),
    ("ko", "en"),
    ("en", "ko"),
    ("zh", "en"),
    ("en", "zh"),
    ("en", "fr"),
    ("fr", "en"),
    ("en", "de"),
    ("de", "en"),
    ("en", "es"),
    ("es", "en"),
}

# Maps human-readable language names to ISO codes used by MarianMT.
_LANG_ALIASES = {
    "english": "en",
    "vietnamese": "vi",
    "japanese": "ja",
    "korean": "ko",
    "chinese": "zh",
    "simplified chinese": "zh",
    "traditional chinese": "zh",
    "french": "fr",
    "german": "de",
    "spanish": "es",
}

# Strip markers used in ``TranslatorService`` prompts so the raw text can be
# extracted when callers use the generic ``generate_text`` interface.
_PROMPT_PREFIX_RE = re.compile(r"^\s*translate this:\s*", re.IGNORECASE)
_PROMPT_SUFFIX_RE = re.compile(r"\s*translation:\s*$", re.IGNORECASE)


def _normalize_lang(value: Optional[str]) -> str:
    if not value:
        return ""
    value = value.strip().lower()
    return _LANG_ALIASES.get(value, value)


class LocalProvider(BaseProvider):
    """Offline MarianMT-based translator. Text-only, no vision support."""

    def __init__(self) -> None:
        super().__init__()
        self._models: Dict[Tuple[str, str], object] = {}
        self._tokenizers: Dict[Tuple[str, str], object] = {}
        self._device: Optional[str] = None
        self._torch = None
        self._AutoTokenizer = None
        self._AutoModel = None
        self._default_src = _normalize_lang(os.getenv("LOCAL_SOURCE_LANG", "en")) or "en"
        self._default_tgt = _normalize_lang(os.getenv("LOCAL_TARGET_LANG", "vi")) or "vi"
        self._load_errors: Dict[Tuple[str, str], str] = {}

    # ------------------------------------------------------------------
    # Lazy import of heavy deps (torch + transformers).
    # ------------------------------------------------------------------
    def _ensure_backend(self) -> bool:
        if self._torch is not None:
            return True
        try:
            import torch  # type: ignore
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

            self._torch = torch
            self._AutoTokenizer = AutoTokenizer
            self._AutoModel = AutoModelForSeq2SeqLM
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            _logger.info("LocalProvider backend ready on %s", self._device)
            return True
        except Exception as exc:  # pragma: no cover - import guard
            _logger.error("LocalProvider requires 'transformers' and 'torch': %s", exc)
            return False

    def _model_name(self, src: str, tgt: str) -> str:
        override = os.getenv(f"LOCAL_MODEL_{src.upper()}_{tgt.upper()}")
        if override:
            return override
        return f"Helsinki-NLP/opus-mt-{src}-{tgt}"

    def _load_pair(self, src: str, tgt: str) -> bool:
        key = (src, tgt)
        if key in self._models:
            return True
        if key in self._load_errors:
            return False
        if not self._ensure_backend():
            return False

        name = self._model_name(src, tgt)
        _logger.info("Loading MarianMT model '%s' ...", name)
        try:
            tokenizer = self._AutoTokenizer.from_pretrained(name)
            model = self._AutoModel.from_pretrained(name, use_safetensors=False).to(self._device)
            model.eval()
            self._tokenizers[key] = tokenizer
            self._models[key] = model
            _logger.info("Loaded %s on %s", name, self._device)
            return True
        except Exception as exc:
            _logger.error("Failed to load %s: %s", name, exc)
            self._load_errors[key] = str(exc)
            return False

    # ------------------------------------------------------------------
    # Translation helpers.
    # ------------------------------------------------------------------
    def _translate_pair(self, text: str, src: str, tgt: str) -> str:
        if not self._load_pair(src, tgt):
            return ""
        tokenizer = self._tokenizers[(src, tgt)]
        model = self._models[(src, tgt)]
        torch = self._torch
        try:
            with torch.no_grad():
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self._device)
                outputs = model.generate(**inputs, max_length=512)
            return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        except Exception as exc:
            _logger.error("Local translation error (%s->%s): %s", src, tgt, exc)
            return ""

    def translate_direct(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate ``text`` from ``src_lang`` to ``tgt_lang``.

        Supports pivoting through English if no direct MarianMT pair exists.
        """
        text = (text or "").strip()
        if not text:
            return ""

        src = _normalize_lang(src_lang) or self._default_src
        tgt = _normalize_lang(tgt_lang) or self._default_tgt

        if src == tgt:
            return text

        # Direct pair.
        if (src, tgt) in _KNOWN_DIRECT_PAIRS or os.getenv(f"LOCAL_MODEL_{src.upper()}_{tgt.upper()}"):
            result = self._translate_pair(text, src, tgt)
            if result:
                self.request_count += 1
                return result

        # Pivot through English if possible.
        if src != "en" and tgt != "en":
            intermediate = self._translate_pair(text, src, "en")
            if intermediate:
                result = self._translate_pair(intermediate, "en", tgt)
                if result:
                    self.request_count += 1
                    return result

        # Last attempt: direct load anyway (in case env override points elsewhere).
        result = self._translate_pair(text, src, tgt)
        if result:
            self.request_count += 1
        return result

    # ------------------------------------------------------------------
    # BaseProvider interface.
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_text(user_prompt: str) -> str:
        if not user_prompt:
            return ""
        cleaned = _PROMPT_PREFIX_RE.sub("", user_prompt)
        cleaned = _PROMPT_SUFFIX_RE.sub("", cleaned)
        return cleaned.strip()

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        stream: bool = False,
    ) -> Union[str, Generator]:
        text = self._extract_text(user_prompt)
        translated = self.translate_direct(text, self._default_src, self._default_tgt)
        if stream:
            def _gen() -> Generator:
                if translated:
                    yield translated
                else:
                    yield ""
            return _gen()
        return translated

    def generate_vision(self, system_prompt: str, user_prompt: str, image_np: np.ndarray) -> str:
        # Vision is intentionally unsupported so callers fall back to OCR + text.
        return ""

    def get_usage_info(self) -> str:
        pairs = ", ".join(f"{s}->{t}" for s, t in self._models.keys()) or "none"
        device = self._device or "not-loaded"
        return (
            f"Local (MarianMT) | Device: {device} | Pairs: {pairs} | "
            f"Requests: {self.request_count} | No API cost"
        )
