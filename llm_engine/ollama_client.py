"""Small, reusable wrappers around locally hosted Ollama models.

Environment variables (required in the project ``.env`` file):
    DEFAULT_OLLAMA_HOST=http://localhost:11434
    DEFAULT_OCR_MODEL=glm-ocr:q8_0
    DEFAULT_QWEN_MODEL=qwen2.5-coder:1.5b

The functions do not pull models. Ollama must already be running and the models
must already be available locally (``ollama list``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from ollama import Client
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class OllamaCallError(RuntimeError):
    """Raised when a local Ollama request cannot be completed."""


def _required_setting(name: str) -> str:
    """Get required Ollama configuration from the environment."""
    value = os.getenv(name)
    if not value:
        raise OllamaCallError(f"Missing required .env setting: {name}")
    return value


def _client() -> Client:
    """Build an Ollama client using the configured local server."""
    return Client(host=_required_setting("DEFAULT_OLLAMA_HOST"))


def _message_content(response: Any) -> str:
    """Return a non-empty message body from an Ollama chat response."""
    message = getattr(response, "message", None)
    content = getattr(message, "content", None)
    if not content or not content.strip():
        raise OllamaCallError("Ollama returned an empty response.")
    return content.strip()


@retry(
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OllamaCallError)),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    stop=stop_after_attempt(3),
    reraise=True,
)
def extract_text_from_image(
    image_path: str | Path,
    *,
    prompt: str | None = None,
) -> str:
    """Extract faithful text from one rendered PDF page using GLM-OCR.

    Parameters
    ----------
    image_path:
        Local path to a PNG/JPEG page image. PDF rendering belongs in the
        document-intelligence layer; this function performs OCR only.
    prompt:
        Optional OCR instruction. The default requests text preservation and
        avoids summaries or invented corrections.

    Returns
    -------
    str
        OCR text for the supplied image. The original image and output should
        be retained by the caller for review and traceability.
    """
    page_image = Path(image_path).expanduser().resolve()
    if not page_image.is_file():
        raise FileNotFoundError(f"OCR image does not exist: {page_image}")

    ocr_prompt = prompt or (
        "Transcribe this document page faithfully. Preserve headings, numbered "
        "lists, bullet lists, table rows, and reading order. Return only the "
        "visible document text; do not summarize, explain, or invent content."
    )
    model = _required_setting("DEFAULT_OCR_MODEL")
    try:
        response = _client().chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": ocr_prompt,
                    "images": [str(page_image)],
                }
            ],
            options={"temperature": 0},
        )
        return _message_content(response)
    except (ConnectionError, TimeoutError, OllamaCallError):
        raise
    except Exception as error:
        raise OllamaCallError(
            "GLM-OCR call failed. Check that Ollama is running and that "
            f"{model!r} is installed."
        ) from error


@retry(
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OllamaCallError)),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    stop=stop_after_attempt(3),
    reraise=True,
)
def ask_qwen(
    prompt: str,
    *,
    system_prompt: str | None = None,
    temperature: float = 0.1,
) -> str:
    """Run a text-only document-intelligence task using local Qwen.

    Use this for later tasks such as structured document profiling, chunking
    recommendations, or quality review. It intentionally does not receive an
    image; image OCR stays in ``extract_text_from_image``.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty")
    if not 0 <= temperature <= 2:
        raise ValueError("temperature must be between 0 and 2")
    model = _required_setting("DEFAULT_QWEN_MODEL")

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt.strip()})

    try:
        response = _client().chat(
            model=model,
            messages=messages,
            options={"temperature": temperature},
        )
        return _message_content(response)
    except (ConnectionError, TimeoutError, OllamaCallError):
        raise
    except Exception as error:
        raise OllamaCallError(
            "Qwen call failed. Check that Ollama is running and that "
            f"{model!r} is installed."
        ) from error
