"""
DashScope Qwen OpenAI-compatible API backend.

This client keeps the same public surface used by the Agent layer while hiding
OpenAI-compatible messages and base64 image conversion inside the backend.
"""

import base64
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import config
from utils.logger import log_error, log_info


class QwenAPIClient:
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **_: Any,
    ):
        api_key = api_key or os.getenv(config.QWEN_API_KEY_ENV)
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set")
        from openai import OpenAI

        self.model_name = model_name or config.QWEN_API_MODEL
        self.base_url = base_url or config.QWEN_API_BASE_URL
        self.timeout = timeout or config.QWEN_API_TIMEOUT
        self.max_new_tokens = max_new_tokens or config.QWEN_API_MAX_TOKENS
        self.temperature = (
            config.QWEN_API_TEMPERATURE if temperature is None else temperature
        )
        self.device = "qwen_api"
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def _image_to_data_url(self, image_path: Union[str, Path]) -> str:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(str(path))

        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def _normalize_image_paths(
        self,
        image_paths: Optional[Union[str, Path, List[Union[str, Path]]]],
    ) -> List[Union[str, Path]]:
        if image_paths is None:
            return []
        if isinstance(image_paths, (str, Path)):
            return [image_paths]
        return list(image_paths)

    def _content_to_openai(self, content: Any) -> List[Dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]

        converted: List[Dict[str, Any]] = []
        for item in content or []:
            if not isinstance(item, dict):
                converted.append({"type": "text", "text": str(item)})
                continue

            item_type = item.get("type")
            if item_type == "text":
                converted.append({"type": "text", "text": item.get("text", "")})
            elif item_type == "image":
                image_source = item.get("image")
                if not image_source:
                    continue
                if isinstance(image_source, str) and image_source.startswith("data:image"):
                    image_url = image_source
                else:
                    image_url = self._image_to_data_url(image_source)
                converted.append({
                    "type": "image_url",
                    "image_url": {"url": image_url},
                })
            elif item_type == "image_url":
                converted.append(item)
            else:
                converted.append({"type": "text", "text": str(item)})

        return converted

    def _messages_to_openai(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        converted = []
        for message in messages:
            role = message.get("role", "user")
            content = self._content_to_openai(message.get("content", ""))
            if role == "system":
                text_parts = [
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                converted.append({
                    "role": role,
                    "content": "\n".join(text_parts),
                })
                continue
            converted.append({
                "role": role,
                "content": content,
            })
        return converted

    def _build_messages(
        self,
        question: str,
        image_paths: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        content: List[Dict[str, Any]] = []
        for image_path in self._normalize_image_paths(image_paths):
            content.append({
                "type": "image_url",
                "image_url": {"url": self._image_to_data_url(image_path)},
            })
        content.append({"type": "text", "text": question})
        messages.append({"role": "user", "content": content})
        return messages

    def _count_images(self, messages: List[Dict[str, Any]]) -> int:
        count = 0
        for message in messages:
            content = message.get("content", [])
            if isinstance(content, list):
                count += sum(
                    1 for item in content
                    if isinstance(item, dict) and item.get("type") == "image_url"
                )
        return count

    def chat(
        self,
        question: Optional[Union[str, List[Dict[str, Any]]]] = None,
        image_paths: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        if question is None and "messages" in kwargs:
            question = kwargs["messages"]
        if question is None:
            raise ValueError("question or messages is required")
        if image_paths is None and "images" in kwargs:
            image_paths = kwargs["images"]

        if isinstance(question, list):
            messages = self._messages_to_openai(question)
        else:
            messages = self._build_messages(question, image_paths, system_prompt)

        image_count = self._count_images(messages)
        start_time = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", self.max_new_tokens),
                temperature=kwargs.get("temperature", self.temperature),
            )
            latency = time.perf_counter() - start_time
            log_info(
                "[QWEN_API_DEBUG] "
                f"backend=qwen_api model={self.model_name} "
                f"image_count={image_count} latency={latency:.2f}"
            )
            content = response.choices[0].message.content
            if isinstance(content, list):
                content = "\n".join(
                    item.get("text", str(item))
                    for item in content
                    if isinstance(item, dict)
                )
            return (content or "").strip()
        except Exception as exc:
            log_error(
                "[QWEN_API_DEBUG] "
                f"model={self.model_name} image_count={image_count} error={exc}"
            )
            raise

    def generate_single_response(
        self,
        text: str,
        images: Optional[List[Union[str, Path]]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        return self.chat(
            question=text,
            image_paths=images or [],
            system_prompt=system_prompt,
        )

    def check_for_zoom_command(self, text: str) -> Optional[tuple]:
        pattern = r"\[ZOOM:\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]"
        match = re.search(pattern, text)
        if not match:
            return None
        try:
            coords = tuple(map(float, match.groups()))
            x, y, w, h = coords
            if all(0 <= val <= 1 for val in coords) and x + w <= 1 and y + h <= 1:
                return coords
        except (ValueError, IndexError):
            return None
        return None

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "backend": "qwen_api",
            "base_url": self.base_url,
            "max_new_tokens": self.max_new_tokens,
        }
