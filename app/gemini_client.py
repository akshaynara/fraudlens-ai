"""
All Gemini access goes through here so the rest of the app never touches the SDK directly.

Auth (either works, picked up automatically by google-genai):
  * AI Studio:  GEMINI_API_KEY=...
  * Vertex AI:  GOOGLE_GENAI_USE_VERTEXAI=true GOOGLE_CLOUD_PROJECT=... GOOGLE_CLOUD_LOCATION=us-central1
                (Cloud Run's service account is used, no key needed)
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable

from pydantic import BaseModel

log = logging.getLogger("fraudlens.gemini")

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover
    genai = None
    types = None


class GeminiClient:
    def __init__(self):
        # 2.5 models are only served to projects that used them recently; new projects should use 3.x.
        self.model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        self.model_pro = os.getenv("GEMINI_MODEL_PRO") or self.model
        self.client = None
        self.available = False
        if genai is None:
            log.warning("google-genai not installed; running in offline fallback mode")
            return
        has_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true", "yes")
        if not (has_key or use_vertex):
            log.warning("No GEMINI_API_KEY / Vertex config; running in offline fallback mode")
            return
        try:
            self.client = genai.Client()
            self.available = True
            log.info("Gemini ready (model=%s, vertex=%s)", self.model, use_vertex)
        except Exception as e:
            log.error("Gemini init failed: %s", e)

    # ---------------------------------------------------------------- building blocks
    @staticmethod
    def file_part(data: bytes, mime: str):
        return types.Part.from_bytes(data=data, mime_type=mime)

    def generate_structured(self, parts: list[Any], schema: type[BaseModel], system: str,
                            model: str | None = None, temperature: float = 0.2) -> BaseModel:
        resp = self.client.models.generate_content(
            model=model or self.model,
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=schema,
                temperature=temperature,
            ),
        )
        if getattr(resp, "parsed", None) is not None:
            return resp.parsed
        return schema.model_validate_json(_strip_fences(resp.text))

    def generate_text(self, prompt: str, system: str, model: str | None = None, temperature: float = 0.4) -> str:
        resp = self.client.models.generate_content(
            model=model or self.model,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system, temperature=temperature),
        )
        return resp.text or ""

    def run_agent(self, prompt: str, system: str, tools: list[Callable], max_calls: int = 14) -> str:
        """Gemini automatic function calling: the model decides which tools to call and when to stop.
        Tool functions log themselves to the case trace, so we get a full investigation record."""
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                tools=tools,
                temperature=0.2,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=max_calls),
            ),
        )
        return resp.text or ""

    def embed(self, texts: list[str], model: str, dims: int) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 50):  # batch politely
            batch = texts[i:i + 50]
            resp = self.client.models.embed_content(
                model=model, contents=batch,
                config=types.EmbedContentConfig(output_dimensionality=dims, task_type="SEMANTIC_SIMILARITY"),
            )
            out.extend([e.values for e in resp.embeddings])
        return out


def _strip_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        t = t.rsplit("```", 1)[0]
    return t.strip()
