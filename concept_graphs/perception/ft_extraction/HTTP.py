"""Image/text features from a RAGMAP ``ragmap-encoder-service`` over HTTP.

Selected with ``ft_extraction=http``. The service owns the encoder (its own
weights, preprocessing and instructions) and answers L2-normalised float32
rows; this side only ships the uint8 RGB crops as PNG and checks every answer
against the identity the service reported at start-up, so a map can never mix
vectors from two encoders or silently receive a short answer.

The wire format mirrors ``ragmap.encoders.wire`` (stdlib + numpy + pillow only,
since this image cannot import ragmap):

* ``GET /healthz`` -> ``{status, profile, model, revision, dim, normalized}``
* ``POST /v1/embed/images`` ``{"images": [base64 PNG]}`` and
  ``POST /v1/embed/texts`` ``{"texts": [...]}`` ->
  ``{profile, revision, dim, count, dtype: "float32", embeddings: base64 LE float32 (n, D)}``

Any error status (507 is GPU OOM on the service) raises; nothing is ever
replaced by zeros.
"""

import base64
import io
import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from .FeatureExtractor import FeatureExtractor

log = logging.getLogger(__name__)


class EncoderServiceError(RuntimeError):
    pass


def _png_b64(img: np.ndarray) -> str:
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) uint8 RGB crop, got {img.dtype} {img.shape}")
    buffer = io.BytesIO()
    # Lossless; level 1 is several times faster to write than the default 6.
    Image.fromarray(np.ascontiguousarray(img), "RGB").save(buffer, format="PNG", compress_level=1)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class HttpFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        base_url: Optional[str],
        device: str = "cuda",
        expected_profile: Optional[str] = None,
        batch_size: int = 256,
        timeout: float = 600.0,
        connect_wait: float = 0.0,
    ):
        base_url = (base_url or "").strip().rstrip("/")
        if not base_url:
            raise EncoderServiceError(
                "ft_extraction=http needs RAGMAP_ENCODER_BASE_URL (a ragmap-encoder-service, "
                "e.g. http://127.0.0.1:8790)."
            )
        self.base_url = base_url
        self.device = device
        self.batch_size = int(batch_size)
        self.timeout = float(timeout)
        self.n_requests = 0
        self.n_images = 0
        self.n_texts = 0
        self.request_seconds = 0.0
        self.identity = self._health(float(connect_wait))
        if expected_profile and self.identity["profile"] != expected_profile:
            raise EncoderServiceError(
                f"{self.base_url} serves encoder profile {self.identity['profile']!r}, "
                f"expected {expected_profile!r}"
            )
        self.dim = int(self.identity["dim"])
        log.info(
            "Encoder service %s: profile=%s model=%s revision=%s dim=%d",
            self.base_url, self.identity["profile"], self.identity.get("model"),
            self.identity["revision"], self.dim,
        )

    # -- transport -----------------------------------------------------------

    def _call(self, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise EncoderServiceError(f"{self.base_url}{path}: HTTP {exc.code}: {detail}") from exc

    def _health(self, connect_wait: float) -> Dict[str, Any]:
        deadline = time.time() + connect_wait
        while True:
            try:
                health = self._call("/healthz")
                break
            except (urllib.error.URLError, OSError) as exc:
                if time.time() >= deadline:
                    raise EncoderServiceError(f"encoder service {self.base_url} unreachable: {exc}") from exc
                time.sleep(2.0)
        if health.get("status") != "ok":
            raise EncoderServiceError(f"encoder service {self.base_url} is not ready: {health}")
        for key in ("profile", "revision", "dim"):
            if health.get(key) in (None, ""):
                raise EncoderServiceError(f"encoder service {self.base_url} /healthz has no {key}: {health}")
        if health.get("normalized") is False:
            raise EncoderServiceError(f"encoder service {self.base_url} does not L2-normalise: {health}")
        return health

    def _embed(self, path: str, key: str, items: List[Any]) -> torch.Tensor:
        parts = [np.zeros((0, self.dim), np.float32)]
        for start in range(0, len(items), self.batch_size):
            chunk = items[start : start + self.batch_size]
            t = time.time()
            answer = self._call(path, {key: chunk})
            self.request_seconds += time.time() - t
            self.n_requests += 1
            for field in ("profile", "revision", "dim"):
                if answer.get(field) != self.identity[field]:
                    raise EncoderServiceError(
                        f"encoder {field} changed under us: {answer.get(field)!r} != {self.identity[field]!r}"
                    )
            if answer.get("count") != len(chunk) or answer.get("dtype") != "float32":
                raise EncoderServiceError(
                    f"bad answer for {len(chunk)} {key}: count={answer.get('count')} dtype={answer.get('dtype')}"
                )
            array = np.frombuffer(base64.b64decode(answer["embeddings"]), dtype="<f4")
            if array.size != len(chunk) * self.dim:
                raise EncoderServiceError(
                    f"embedding payload has {array.size} floats, expected {len(chunk)} x {self.dim}"
                )
            parts.append(array.reshape(len(chunk), self.dim).astype(np.float32))
        features = np.concatenate(parts, axis=0)
        if not np.isfinite(features).all():
            raise EncoderServiceError(f"{self.base_url}{path} returned non-finite embeddings")
        return torch.from_numpy(features).to(self.device)

    # -- FeatureExtractor ------------------------------------------------------

    def __call__(self, imgs: List[np.ndarray]) -> torch.Tensor:
        self.n_images += len(imgs)
        return self._embed("/v1/embed/images", "images", [_png_b64(img) for img in imgs])

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        self.n_texts += len(texts)
        return self._embed("/v1/embed/texts", "texts", [str(text) for text in texts])

    def describe(self) -> Dict[str, Any]:
        """The encoder this map was built with, for run.json."""

        return {
            "mode": "http",
            "base_url": self.base_url,
            **{k: self.identity.get(k) for k in ("profile", "model", "revision", "dim", "normalized")},
            "requests": self.n_requests,
            "images": self.n_images,
            "texts": self.n_texts,
            "request_seconds": round(self.request_seconds, 3),
        }
