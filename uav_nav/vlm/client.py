"""
Ollama HTTP client wrapper for BakLLaVA/LLaVA inference.

Runs 100% offline — Ollama serves the model locally.
Implements graceful degradation: if Ollama is unreachable, returns None
so the UKF continues predicting without VLM updates.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Optional, Union

from uav_nav.config import get_config
from uav_nav.vlm.parser import parse_vlm_response
from uav_nav.vlm.prompt_builder import build_messages
from uav_nav.vlm.schemas import SemanticObservation

log = logging.getLogger(__name__)


class VLMUnavailableError(Exception):
    pass


def _encode_image(image_source: Union[str, Path, bytes]) -> str:
    """Return base64-encoded image string from file path or raw bytes."""
    if isinstance(image_source, bytes):
        return base64.b64encode(image_source).decode("utf-8")
    path = Path(image_source)
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class VLMClient:
    """
    Queries BakLLaVA (or any LLaVA-family model) via the local Ollama server.

    Args:
        model: Ollama model name (default from config).
        host:  Ollama HTTP base URL (default from config).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
    ):
        cfg = get_config()
        self.model = model or cfg.ollama_model
        self.host = host or cfg.ollama_host
        self.timeout = cfg.vlm_timeout_s
        self._client = None  # lazy import of ollama

    def _get_client(self):
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self.host)
            except ImportError:
                raise VLMUnavailableError(
                    "ollama package not installed. Run: pip install ollama"
                )
        return self._client

    def is_available(self) -> bool:
        """Check whether Ollama server is reachable."""
        try:
            client = self._get_client()
            client.list()  # lightweight ping
            return True
        except Exception:
            return False

    def query(
        self,
        image_source: Union[str, Path, bytes],
    ) -> Optional[SemanticObservation]:
        """
        Send image to VLM, parse structured SemanticObservation.

        Args:
            image_source: File path (str/Path) or raw image bytes.

        Returns:
            SemanticObservation if successful, None on any failure.
        """
        try:
            b64 = _encode_image(image_source)
        except Exception as exc:
            log.warning("Failed to encode image: %s", exc)
            return None

        messages = build_messages(b64)

        try:
            client = self._get_client()
            response = client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": 0.1},  # low temp for deterministic JSON
            )
            raw_text = response["message"]["content"]
            log.debug("VLM raw response: %.500s", raw_text)
        except VLMUnavailableError:
            raise
        except Exception as exc:
            log.warning("Ollama query failed: %s", exc)
            return None

        obs = parse_vlm_response(raw_text)
        if obs is None:
            log.warning("VLM returned unparseable response.")
        return obs
