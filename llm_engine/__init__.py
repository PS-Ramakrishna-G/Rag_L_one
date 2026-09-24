"""Reusable local LLM helpers for the document-intelligence pipeline."""

from .ollama_client import ask_qwen, extract_text_from_image

__all__ = ["ask_qwen", "extract_text_from_image"]
