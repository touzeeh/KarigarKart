import asyncio
import base64
import json
import logging
import os
import time
from io import BytesIO
from typing import Any

import httpx
from PIL import Image

LOGGER = logging.getLogger("karigarkart.ai")
MAX_IMAGE_BYTES = 1_500_000

class AIServiceError(Exception):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

class GeminiAIService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").strip()
        self.retries = max(0, min(4, int(os.getenv("AI_MAX_RETRIES", "3"))))
        timeout = float(os.getenv("AI_TIMEOUT_SECONDS", "45"))
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10, write=20, pool=10))

    async def close(self):
        await self.client.aclose()

    async def generate(self, prompt: str, image: bytes | None = None, image_mime: str = "image/jpeg", audio: bytes | None = None, audio_mime: str = "audio/m4a", max_tokens: int = 700) -> str:
        if not self.api_key:
            raise AIServiceError("AI service is not configured on the server.", 503)
        prompt = prompt.strip()
        if not prompt:
            raise AIServiceError("AI prompt cannot be empty.", 400)
        parts: list[dict[str, Any]] = [{"text": prompt}]
        if image is not None:
            parts.append({"inline_data": {"mime_type": image_mime, "data": base64.b64encode(image).decode("ascii")}})
        if audio is not None:
            parts.append({"inline_data": {"mime_type": audio_mime, "data": base64.b64encode(audio).decode("ascii")}})
        body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": {"temperature": 0.2, "maxOutputTokens": max(64, min(max_tokens, 1200)), "responseMimeType": "application/json", "thinkingConfig": {"thinkingBudget": 0}}}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        started = time.perf_counter()
        for attempt in range(self.retries + 1):
            try:
                response = await self.client.post(url, headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"}, json=body)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt >= self.retries:
                    LOGGER.exception("AI network/timeout failure")
                    raise AIServiceError("The AI service could not be reached. Please try again.", 504) from exc
                await asyncio.sleep(min(8.0, 0.75 * (2 ** attempt)))
                continue
            if response.status_code == 200:
                LOGGER.info("AI success model=%s elapsed_ms=%d", self.model, int((time.perf_counter() - started) * 1000))
                return self._extract(response)
            retryable = response.status_code == 429 or 500 <= response.status_code < 600
            LOGGER.warning("AI provider status=%s attempt=%d detail=%s", response.status_code, attempt + 1, self._error(response))
            if retryable and attempt < self.retries:
                try:
                    delay = float(response.headers.get("retry-after", ""))
                except ValueError:
                    delay = 0.75 * (2 ** attempt)
                await asyncio.sleep(min(8.0, max(0.5, delay)))
                continue
            if response.status_code == 429:
                raise AIServiceError("The AI service is temporarily busy. Please try again.", 503)
            if 500 <= response.status_code < 600:
                raise AIServiceError("The AI service is temporarily unavailable. Please try again.", 503)
            raise AIServiceError("The AI request was rejected. Please check the input and try again.", 502)
        raise AIServiceError("The AI request failed.", 502)

    @staticmethod
    def _error(response):
        try:
            return str(response.json().get("error", {}).get("message", "provider error"))[:400]
        except (ValueError, TypeError, AttributeError):
            return response.text[:400]

    @staticmethod
    def _extract(response):
        try:
            data = response.json()
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(str(part.get("text", "")) for part in parts).strip()
            if not text:
                raise AIServiceError("AI returned an empty response.", 502)
            return text
        except AIServiceError:
            raise
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            raise AIServiceError("AI returned an invalid response.", 502) from exc

def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text[3:].strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AIServiceError("AI returned malformed structured data.", 502) from exc
    if not isinstance(value, dict):
        raise AIServiceError("AI returned an unexpected response shape.", 502)
    return value

def compact_image(data: bytes, mime: str):
    if len(data) <= MAX_IMAGE_BYTES:
        return data, mime
    try:
        with Image.open(BytesIO(data)) as image:
            image = image.convert("RGB")
            image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
            output = BytesIO()
            image.save(output, format="JPEG", quality=82, optimize=True)
            data = output.getvalue()
    except Exception as exc:
        raise AIServiceError("The image could not be compressed for AI processing.", 413) from exc
    if len(data) > MAX_IMAGE_BYTES:
        raise AIServiceError("The image is too large to process. Please choose a smaller image.", 413)
    return data, "image/jpeg"
