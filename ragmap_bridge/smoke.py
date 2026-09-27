"""`ragmap-smoke`: end-to-end check of `ragmap-run` on a synthetic 5-frame scene.

Runs inside the image, CPU-only, with no network access to any VLM:

* the scene is ultralytics' bundled `bus.jpg` seen five times from slightly
  translated cameras, over a flat synthetic depth of 4 m, so YOLO-World has
  something real to detect;
* captioning goes to an in-process fake server that enforces the same request
  rules as RAGMAP's `ragmap-vlm-service` (user-only messages, `model` must be
  the served profile, images as `data:` URLs, greedy sampling), so a request
  shape the real service would 400 fails here too;
* the output is validated against the object_mapping contract, including that
  coordinates are in the input world frame.

Uses the small ViT-B-32 CLIP (ft_extraction=CLIP) instead of ViT-H-14 to keep
the CI download small; pass `--clip h14` to exercise the production encoder.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

FAKE_PROFILE = "fake_profile"
CAMERA_ORIGIN = np.array([1.0, 2.0, 0.5])
DEPTH_M = 4.0


def make_scene(root: Path, n_frames: int = 5) -> None:
    import cv2
    from ultralytics.utils import ASSETS

    rgb = cv2.imread(str(Path(ASSETS) / "bus.jpg"))
    rgb = cv2.resize(rgb, (480, 640), interpolation=cv2.INTER_AREA)
    h, w = rgb.shape[:2]
    depth_scale = 1000.0
    depth = np.full((h, w), int(DEPTH_M * depth_scale), dtype=np.uint16)

    (root / "rgb").mkdir(parents=True, exist_ok=True)
    (root / "depth").mkdir(parents=True, exist_ok=True)
    meta = {
        "fx": 500.0, "fy": 500.0, "cx": (w - 1) / 2, "cy": (h - 1) / 2,
        "width": w, "height": h, "depth_scale": depth_scale,
        "up_axis": "-y", "frame": "ragmap_scene",
    }
    (root / "meta.json").write_text(json.dumps(meta))
    with open(root / "frames.jsonl", "w") as f:
        for i in range(n_frames):
            cv2.imwrite(str(root / "rgb" / f"{i:06d}.jpg"), rgb)
            cv2.imwrite(str(root / "depth" / f"{i:06d}.png"), depth)
            pose = np.eye(4)
            pose[:3, 3] = CAMERA_ORIGIN + np.array([0.01 * i, 0.0, 0.0])
            f.write(json.dumps({
                "frame_index": 10 + i,  # deliberately not 0-based
                "observation_id": f"obs-{i}",
                "rgb": f"rgb/{i:06d}.jpg",
                "depth": f"depth/{i:06d}.png",
                "pose": pose.reshape(-1).tolist(),
            }) + "\n")


class FakeVlm(BaseHTTPRequestHandler):
    """Mimics the request validation of ragmap.vlm.service.prepare_request."""

    requests = []
    errors = []

    def log_message(self, *args):  # quiet
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/status":
            return self._send(200, {"profile": FAKE_PROFILE, "loaded": True})
        return self._send(404, {"error": {"message": "not found"}})

    def _reject(self, message: str):
        FakeVlm.errors.append(message)
        return self._send(400, {"error": {"message": message}})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self._reject(f"unexpected path {self.path}")
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeVlm.requests.append(body)
        model = str(body.get("model") or "")
        if model and model != FAKE_PROFILE:
            return self._reject(f"service holds {FAKE_PROFILE!r}, not {model!r}")
        for name in ("temperature", "top_p"):
            if body.get(name) is not None and float(body[name]) not in (0.0, 1.0):
                return self._reject(f"{name} not greedy")
        if body.get("stream"):
            return self._reject("stream")
        text, images = "", 0
        for message in body.get("messages") or []:
            if message.get("role") != "user":
                return self._reject(f"only 'user' messages; received {message.get('role')!r}")
            for block in message["content"]:
                if block["type"] == "text":
                    text += block["text"]
                elif block["type"] == "image_url":
                    url = block["image_url"]["url"]
                    if not url.startswith("data:image/"):
                        return self._reject("image is not a data: URL")
                    base64.b64decode(url.partition(",")[2], validate=True)
                    images += 1
                else:
                    return self._reject(f"block type {block['type']}")
        if images == 0:
            return self._reject("A request needs at least one image")
        if "single word" in text:
            answer = "The best description of the central object is bus."
        else:
            answer = "The central object is a yellow city bus."
        return self._send(200, {
            "id": "chatcmpl-fake", "object": "chat.completion", "created": 0, "model": FAKE_PROFILE,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        })


REQUIRED = {
    "id": str, "label": str, "caption": (str, type(None)), "crop": (str, type(None)),
    "centroid": list, "bbox_min": list, "bbox_max": list, "pointcloud": str,
    "frame_indices": list, "floor": type(None), "room": type(None), "extra": dict,
}


def validate(output: Path, captioned: bool) -> list:
    lines = (output / "objects.jsonl").read_text().splitlines()
    objects = [json.loads(line) for line in lines if line.strip()]
    assert objects, "no objects were produced from the bus scene"
    ids = set()
    for obj in objects:
        for key, kind in REQUIRED.items():
            assert key in obj, f"{obj.get('id')}: missing {key}"
            assert isinstance(obj[key], kind), f"{obj['id']}: {key} has type {type(obj[key]).__name__}"
        assert obj["id"] not in ids, f"duplicate id {obj['id']}"
        ids.add(obj["id"])
        lo, hi, c = (np.array(obj[k]) for k in ("bbox_min", "bbox_max", "centroid"))
        assert lo.shape == hi.shape == c.shape == (3,)
        assert np.all(lo <= c + 1e-6) and np.all(c <= hi + 1e-6), f"{obj['id']}: centroid outside bbox"
        # World frame: the depth plane is DEPTH_M in front of cameras at CAMERA_ORIGIN.
        assert abs(c[2] - (CAMERA_ORIGIN[2] + DEPTH_M)) < 0.25, f"{obj['id']}: z={c[2]} not in world frame"
        assert (output / obj["pointcloud"]).is_file(), obj["pointcloud"]
        if obj["crop"]:
            assert (output / obj["crop"]).is_file(), obj["crop"]
        assert obj["frame_indices"] and set(obj["frame_indices"]) <= set(range(10, 15)), obj["frame_indices"]
        if captioned:
            assert obj["label"] == "bus", f"label {obj['label']!r} did not come from the VLM tag"
            assert obj["caption"] == "yellow city bus", f"caption {obj['caption']!r}"
        else:
            assert obj["extra"]["label_source"] == "detector", obj["extra"]
    run = json.loads((output / "run.json").read_text())
    assert run["counts"]["n_objects"] == len(objects)
    assert run["counts"]["input_frames"] == 5
    return objects


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ragmap-smoke")
    parser.add_argument("--clip", choices=("b32", "h14"), default="b32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workdir", type=Path, default=None)
    args = parser.parse_args(argv)

    from ragmap_bridge.run import main as ragmap_run

    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="ragmap-smoke-"))
    scene, output = workdir / "input", workdir / "output"
    make_scene(scene)

    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeVlm)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    import os

    os.environ["RAGMAP_VLM_BASE_URL"] = f"http://127.0.0.1:{server.server_address[1]}"
    os.environ.pop("RAGMAP_VLM_MODEL", None)  # exercise /status discovery

    overrides = ["final_min_segments=1", "dataloader.num_workers=0"]
    if args.clip == "b32":
        overrides.append("ft_extraction=CLIP")

    common = ["--input", str(scene), "--device", args.device]
    # 1) detector labels only
    assert ragmap_run(common + ["--output", str(output / "plain"), *overrides]) == 0
    plain = validate(output / "plain", captioned=False)
    # 2) captions + tags from the (fake) local VLM
    assert ragmap_run(common + ["--output", str(output / "captioned"), "--caption", *overrides]) == 0
    captioned = validate(output / "captioned", captioned=True)
    server.shutdown()

    assert not FakeVlm.errors, f"fake VLM rejected requests: {FakeVlm.errors[:3]}"
    assert FakeVlm.requests, "no VLM requests were made"
    assert all(r.get("model") == FAKE_PROFILE for r in FakeVlm.requests)

    print(json.dumps({
        "ok": True,
        "workdir": str(workdir),
        "plain_objects": [(o["id"], o["label"], o["frame_indices"]) for o in plain],
        "captioned_objects": [(o["id"], o["label"], o["caption"]) for o in captioned],
        "vlm_requests": len(FakeVlm.requests),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
