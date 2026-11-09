"""
Vision-language model client for the "upload a dashboard/camera screenshot and ask
about it" mode. Uses Google Gemini's free-tier vision model — a local VLM (LLaVA,
Qwen-VL) is not realistic on a 4GB-VRAM laptop, so this offloads the heavy
multimodal inference to a free hosted API, same reasoning as the text LLM.
"""
import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")

VLM_SYSTEM_INSTRUCTION = """You are an automotive engineering assistant analyzing a
vehicle camera or dashboard image. Describe only what is visibly present: road
conditions, lane markings, visible warning indicators, nearby vehicles/pedestrians,
or dashboard readouts. Do not speculate about mechanical causes you cannot see. If
retrieved engineering context is provided alongside the image, ground your
explanation in that context and cite the source document."""


class VLMNotConfigured(Exception):
    """Raised when no GOOGLE_API_KEY is set. VLM is an optional add-on module —
    the core RAG system (retrieval + Groq text generation) works fully without it."""
    pass


def vlm_available() -> bool:
    return bool(os.environ.get("GOOGLE_API_KEY"))


def analyze_image(image_bytes: bytes, question: str, context: str | None = None) -> str:
    if not vlm_available():
        raise VLMNotConfigured(
            "VLM mode is optional and not configured. The core RAG system (text "
            "questions over your documents, via Groq) works fully without it. To "
            "enable image analysis later, get a free Google AI Studio key at "
            "https://aistudio.google.com/apikey — note: as of March 2026 some "
            "regions are prompted for prepaid billing setup on new accounts, so "
            "confirm it's genuinely free for your account before relying on it. A "
            "no-card alternative (e.g. a free Hugging Face vision endpoint) can be "
            "swapped in here instead if Google asks for billing."
        )

    import google.generativeai as genai
    from PIL import Image
    import io

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=VLM_SYSTEM_INSTRUCTION)

    image = Image.open(io.BytesIO(image_bytes))
    prompt_parts = [image, question]
    if context:
        prompt_parts.append(f"\n\nRelevant retrieved documentation:\n{context}")

    resp = model.generate_content(prompt_parts)
    return resp.text
