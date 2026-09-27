import os
import cv2
import base64
import openai
import numpy as np
from typing import Optional
from .ImageCaptioner import ImageCaptioner
import logging

log = logging.getLogger(__name__)


def normalize_base_url(base_url: Optional[str]) -> Optional[str]:
    """Accept `http://host:port` or `http://host:port/v1` for an OpenAI-compatible server.

    The openai SDK appends `/chat/completions` to its base URL, so the `/v1`
    prefix must be part of it.
    """
    if not base_url:
        return None
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"
    return base_url


class OpenAICaptioner(ImageCaptioner):
    """Caption/tag objects through an OpenAI-compatible chat-completions endpoint.

    With `base_url` unset this is the upstream behaviour (api.openai.com, needs
    OPENAI_API_KEY). With `base_url` set (conf default: env RAGMAP_VLM_BASE_URL)
    it talks to a local server instead -- RAGMAP's `ragmap-vlm-service` or vLLM.

    `fold_system_prompt`: RAGMAP's service only accepts `user` messages (a
    `system` message is a 400), so the system prompt is prepended to the user
    text instead. `None` means "fold iff base_url is set".
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        fold_system_prompt: Optional[bool] = None,
        max_tokens: Optional[int] = 64,
        timeout: float = 600.0,
        max_retries: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model or ""
        self.base_url = normalize_base_url(base_url)
        self.fold_system_prompt = (
            bool(self.base_url) if fold_system_prompt is None else bool(fold_system_prompt)
        )
        self.max_tokens = max_tokens
        self.n_failures = 0
        self.last_error = None
        client_kwargs = dict(timeout=timeout, max_retries=max_retries)
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
            # Local servers ignore the key, but the SDK refuses to start without one.
            client_kwargs["api_key"] = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        elif api_key:
            client_kwargs["api_key"] = api_key
        self.client = openai.OpenAI(**client_kwargs)
        log.info(
            "OpenAICaptioner endpoint=%s model=%r fold_system_prompt=%s",
            self.base_url or "api.openai.com",
            self.model,
            self.fold_system_prompt,
        )

    def encode_images(self, images: [np.ndarray]) -> [str]:
        # To RGB
        images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
        return [
            base64.b64encode(cv2.imencode(".jpg", img)[1]).decode("utf-8")
            for img in images
        ]

    def build_messages(self, base64_images: [str]) -> list:
        image_blocks = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
            }
            for base64_img in base64_images
        ]
        if self.fold_system_prompt:
            text = self.full_system_prompt + "\n\n" + self.user_query
            return [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": text}, *image_blocks],
                }
            ]
        return [
            {"role": "system", "content": self.full_system_prompt},
            {
                "role": "user",
                "content": [{"type": "text", "text": self.user_query}, *image_blocks],
            },
        ]

    def __call__(self, images: [np.ndarray]) -> str:
        if len(images) > self.max_images:
            images = images[: self.max_images]

        base64_images = self.encode_images(images)
        messages = self.build_messages(base64_images)
        request = dict(model=self.model, messages=messages, temperature=0.0)
        if self.max_tokens:
            request["max_tokens"] = int(self.max_tokens)
        try:
            response = self.client.chat.completions.create(**request)
            response = response.choices[0].message.content or ""
        except Exception as e:
            self.n_failures += 1
            self.last_error = f"{type(e).__name__}: {e}"
            log.warning("Error: Could not generate caption (%s)", self.last_error)
            response = "Invalid"

        return self.postprocess_response(response.strip())
