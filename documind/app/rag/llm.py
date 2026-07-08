"""Gemini calls (REST) with model fallback + token usage."""
import base64
import requests
from app import config


def _endpoint(model: str) -> str:
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={config.GEMINI_API_KEY}"


def generate(system: str, user_text: str) -> tuple[str, int, int]:
    """Return (text, prompt_tokens, output_tokens). Tries models in order."""
    last = "no models"
    for model in config.GEMINI_MODELS:
        gen: dict = {"maxOutputTokens": 900, "temperature": 0.3}
        if model.startswith("gemini-2.5"):
            gen["thinkingConfig"] = {"thinkingBudget": 0}
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "generationConfig": gen,
        }
        try:
            r = requests.post(_endpoint(model), json=body, timeout=90)
            if not r.ok:
                last = f"{model} {r.status_code}"
                continue
            d = r.json()
            cand = d.get("candidates", [{}])[0]
            text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
            um = d.get("usageMetadata", {})
            if text:
                return text, um.get("promptTokenCount", 0), um.get("candidatesTokenCount", 0)
            last = f"{model} empty"
        except Exception as e:
            last = f"{model} {e}"
    raise RuntimeError("gemini failed: " + last)


def image_to_text(image_bytes: bytes, mime: str) -> tuple[str, int, int]:
    """Extract text + describe an image via Gemini vision."""
    b64 = base64.b64encode(image_bytes).decode()
    body = {
        "contents": [{"role": "user", "parts": [
            {"text": "Extract ALL text visible in this image verbatim. Then add a one-line description of any charts, tables, diagrams, or key visual content. Be thorough and factual."},
            {"inlineData": {"mimeType": mime, "data": b64}},
        ]}],
        "generationConfig": {"maxOutputTokens": 1200, "temperature": 0.1},
    }
    r = requests.post(_endpoint(config.GEMINI_VISION_MODEL), json=body, timeout=120)
    r.raise_for_status()
    d = r.json()
    cand = d.get("candidates", [{}])[0]
    text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
    um = d.get("usageMetadata", {})
    return text, um.get("promptTokenCount", 0), um.get("candidatesTokenCount", 0)
