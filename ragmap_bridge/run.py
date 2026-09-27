"""`ragmap-run --input /input --output /output [--caption]`

Runs concept-nodes (ConceptGraphs, default algo=CGDetector) on a RAGMAP scene
export and writes objects in the shared object_mapping contract.

INPUT  /input/meta.json, /input/frames.jsonl (+ the rgb/depth files they name)
OUTPUT /output/objects.jsonl, objects/<id>.ply, crops/<id>.png,
       clip_features.npy, run.json

Every coordinate written is in the input world frame: object point clouds are
back-projected with the frame's camera-to-world pose and never re-anchored
(relative_pose is forced off).

Captioning (`--caption`) turns on concept-nodes' caption *and* tag passes and
sends them to a local OpenAI-compatible server only: RAGMAP_VLM_BASE_URL must
be set (e.g. http://127.0.0.1:PORT) and RAGMAP_VLM_MODEL names the model
(for ragmap-vlm-service: its profile, e.g. qwen3_vl_8b). The hosted OpenAI API
is never used by this entry point.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import platform
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
CONF_DIR = REPO_ROOT / "conf"
SCHEMA = "ragmap.object_mapping/v1"

log = logging.getLogger("ragmap_bridge")

#: Captioner answers that are not a usable label/caption.
_INVALID_ANSWERS = {"", "invalid", "empty", "background"}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ragmap-run",
        description="concept-nodes object mapping over a RAGMAP scene export.",
    )
    parser.add_argument("--input", required=True, type=Path, help="Directory with meta.json + frames.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="Directory to write objects.jsonl etc.")
    parser.add_argument(
        "--caption", action="store_true",
        help="Caption and tag objects with the local VLM at RAGMAP_VLM_BASE_URL.",
    )
    parser.add_argument("--algo", default="CGDetector", help="concept-nodes algo config (default CGDetector)")
    parser.add_argument(
        "--device", default="cuda",
        help="Torch device. Default cuda, and it fails if CUDA is unavailable rather than silently using CPU.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth frame (dataset.sequence_stride)")
    parser.add_argument("--max-frames", type=int, default=-1, help="Stop after this many frames (before stride)")
    parser.add_argument(
        "--weights-dir", type=Path,
        default=Path(os.environ.get("CONCEPT_NODES_WEIGHTS", "/opt/weights")),
        help="Directory holding mobile_sam.pt, yolov8s-world.pt, scannet200_classes.txt",
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="Extra hydra overrides for concept-nodes, e.g. sim_thresh=0.89 ft_extraction=CLIP",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Local VLM endpoint
# ---------------------------------------------------------------------------


def _get_json(url: str, timeout: float = 10.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def resolve_vlm() -> Dict[str, str]:
    """Base URL and model for the local captioner, or exit with a clear reason."""

    base_url = (os.environ.get("RAGMAP_VLM_BASE_URL") or "").strip().rstrip("/")
    if not base_url:
        raise SystemExit(
            "--caption needs RAGMAP_VLM_BASE_URL (an OpenAI-compatible local server, "
            "e.g. RAGMAP's ragmap-vlm-service). Hosted captioners are not used."
        )
    root = base_url[: -len("/v1")] if base_url.endswith("/v1") else base_url
    model = (os.environ.get("RAGMAP_VLM_MODEL") or "").strip()
    source = "env"
    if not model:
        # ragmap-vlm-service: GET /status -> {"profile": ...}. vLLM: GET /v1/models.
        for probe, extract in (
            (root + "/status", lambda d: d.get("profile")),
            (root + "/v1/models", lambda d: (d.get("data") or [{}])[0].get("id")),
        ):
            try:
                model = str(extract(_get_json(probe)) or "")
            except (urllib.error.URLError, OSError, ValueError, AttributeError, IndexError):
                model = ""
            if model:
                source = probe
                break
    if not model:
        raise SystemExit(
            f"Could not determine the VLM model: RAGMAP_VLM_MODEL is unset and neither "
            f"{root}/status nor {root}/v1/models named one."
        )
    os.environ["RAGMAP_VLM_BASE_URL"] = base_url
    os.environ["RAGMAP_VLM_MODEL"] = model
    return {"base_url": base_url, "model": model, "model_source": source}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def compose_config(args: argparse.Namespace, n_input_frames: int):
    from hydra import compose, initialize_config_dir

    output = args.output.resolve()
    overrides = [
        f"algo={args.algo}",
        "dataset=RagmapScene",
        f"dataset.base_path={args.input.resolve()}",
        f"dataset.sequence_stride={args.stride}",
        "dataset.relative_pose=false",
        f"cache_dir={args.weights_dir}",
        f"data_dir={args.input.resolve()}",
        f"output_dir={output}",
        f"device={args.device}",
        f"caption={'true' if args.caption else 'false'}",
        f"tag={'true' if args.caption else 'false'}",
        "save_map=false",
        # Default Hugging Face cache (HF_HOME) instead of the checkpoint dir, so
        # the ViT-H-14 weights come from the mounted HF cache.
        "ft_extraction.cache_dir=null",
    ]
    if args.max_frames and args.max_frames > 0:
        overrides.append(f"dataset.sequence_end={min(args.max_frames, n_input_frames)}")
    overrides += list(args.overrides)
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        cfg = compose(config_name="main", overrides=overrides)
    return cfg, overrides


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _clean_answer(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = str(text).strip()
    if text.lower() in _INVALID_ANSWERS:
        return None
    return text


def export_objects(main_map, output: Path, captioned: bool) -> Dict[str, Any]:
    import cv2
    import numpy as np
    import open3d as o3d

    objects_dir = output / "objects"
    crops_dir = output / "crops"
    objects_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    records = []
    features = []
    counts = {"labels_from_vlm": 0, "labels_from_detector": 0, "labels_fallback": 0, "vlm_background": 0}
    for i, obj in enumerate(main_map):
        object_id = f"obj{i:04d}"
        points = np.asarray(obj.pcd.points)
        detector_label = obj.detector_label
        tag = _clean_answer(obj.tag) if captioned else None
        caption = _clean_answer(obj.caption) if captioned else None
        vlm_background = captioned and str(obj.tag).strip().lower() == "background"

        if tag:
            label, label_source = tag, "vlm_tag"
            counts["labels_from_vlm"] += 1
        elif detector_label:
            label, label_source = detector_label, "detector"
            counts["labels_from_detector"] += 1
        else:
            label, label_source = "object", "fallback"
            counts["labels_fallback"] += 1
        counts["vlm_background"] += int(vlm_background)

        ply_rel = f"objects/{object_id}.ply"
        o3d.io.write_point_cloud(str(output / ply_rel), obj.pcd)

        crop_rel = None
        crop_frame_index = None
        segments = obj.segments.get_sorted()
        if segments:
            best = segments[0]
            crop_rel = f"crops/{object_id}.png"
            cv2.imwrite(str(output / crop_rel), cv2.cvtColor(best.rgb, cv2.COLOR_RGB2BGR))
            crop_frame_index = getattr(best, "frame_index", None)

        features.append(np.asarray(obj.semantic_ft, dtype=np.float32).reshape(-1))
        record = {
            "id": object_id,
            "label": label,
            "caption": caption,
            "crop": crop_rel,
            "centroid": points.mean(axis=0).tolist(),
            "bbox_min": points.min(axis=0).tolist(),
            "bbox_max": points.max(axis=0).tolist(),
            "pointcloud": ply_rel,
            "frame_indices": sorted(int(f) for f in getattr(obj, "frame_indices", set())),
            "floor": None,
            "room": None,
            "extra": {
                "source": "concept-nodes",
                "label_source": label_source,
                "detector_label": detector_label,
                "detector_label_votes": dict(getattr(obj, "label_votes", {}) or {}),
                "vlm_tag": obj.tag if captioned else None,
                "vlm_caption_raw": obj.caption if captioned else None,
                "vlm_background": vlm_background,
                "n_detections": int(obj.n_segments),
                "n_points": int(len(points)),
                "clip_feature_row": i,
                "crop_frame_index": crop_frame_index,
            },
        }
        records.append(record)

    with open(output / "objects.jsonl", "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    if features:
        np.save(output / "clip_features.npy", np.stack(features, axis=0))
    return {"n_objects": len(records), **counts, "clip_feature_dim": int(features[0].shape[0]) if features else None}


def _versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {"python": platform.python_version()}
    for name in ("torch", "torchvision", "open3d", "open_clip", "ultralytics", "openai", "numpy"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", None)
        except Exception:  # noqa: BLE001
            versions[name] = None
    try:
        import torch

        versions["torch_cuda"] = torch.version.cuda
    except Exception:  # noqa: BLE001
        pass
    return versions


def _prepare_output(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    # A rerun into the same directory must not leave stale objects behind.
    for name in ("objects", "crops"):
        shutil.rmtree(output / name, ignore_errors=True)
    for name in ("objects.jsonl", "clip_features.npy", "run.json"):
        (output / name).unlink(missing_ok=True)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr
    )
    args = parse_args(argv)
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.time()

    if not (args.input / "meta.json").is_file() or not (args.input / "frames.jsonl").is_file():
        raise SystemExit(f"{args.input} must contain meta.json and frames.jsonl")
    for name in ("mobile_sam.pt", "yolov8s-world.pt", "scannet200_classes.txt"):
        if args.algo == "CGDetector" and not (args.weights_dir / name).is_file():
            raise SystemExit(f"Missing checkpoint {args.weights_dir / name}")

    import torch
    from omegaconf import OmegaConf

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available in this container (was a GPU passed with "
            "--device nvidia.com/gpu=...?). Use --device cpu to run on CPU deliberately."
        )

    vlm = resolve_vlm() if args.caption else None

    sys.path.insert(0, str(REPO_ROOT))
    from concept_graphs.datasets.RagmapScene import read_frames, read_meta
    from main import build_map
    import hydra

    meta = read_meta(args.input)
    n_input_frames = len(read_frames(args.input))
    cfg, overrides = compose_config(args, n_input_frames)
    _prepare_output(args.output)
    timings: Dict[str, float] = {}

    dataset, main_map, map_stats = build_map(cfg)
    timings.update(
        model_and_data_load=map_stats["load_time"],
        mapping_loop=map_stats["loop_time"],
        postprocess=map_stats["postprocess_time"],
    )

    caption_info: Dict[str, Any] = {"enabled": bool(args.caption)}
    if args.caption and len(main_map):
        caption_info.update(vlm or {})
        t = time.time()
        captioner = hydra.utils.instantiate(cfg.vlm_caption)
        captioner.caption_map(main_map)
        tagger = hydra.utils.instantiate(cfg.vlm_tag)
        tagger.caption_map(main_map)
        timings["caption_and_tag"] = time.time() - t
        caption_info.update(
            caption_failures=captioner.n_failures,
            tag_failures=tagger.n_failures,
            last_error=tagger.last_error or captioner.last_error,
        )
        if len(main_map) and captioner.n_failures == len(main_map) and tagger.n_failures == len(main_map):
            raise SystemExit(
                f"Every caption and tag request to {vlm['base_url']} failed: "
                f"{tagger.last_error or captioner.last_error}"
            )
    elif args.caption:
        caption_info.update(vlm or {})

    t = time.time()
    export_counts = export_objects(main_map, args.output, captioned=bool(args.caption))
    timings["export"] = time.time() - t
    timings["total"] = time.time() - t0

    run = {
        "schema": SCHEMA,
        "tool": "concept-nodes",
        "algo": args.algo,
        "started_at": started_at,
        "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": os.environ.get("CONCEPT_NODES_GIT_SHA"),
        "image": os.environ.get("CONCEPT_NODES_IMAGE"),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "hydra_overrides": overrides,
        "meta": meta,
        "coordinate_frame": meta.get("frame"),
        "counts": {
            "input_frames": n_input_frames,
            "processed_frames": map_stats["n_frames"],
            "frames_with_segments": map_stats["n_frames_with_segments"],
            "detected_segments": map_stats["n_detected_segments"],
            **export_counts,
        },
        "timings_seconds": timings,
        "fps": map_stats["fps"],
        "caption": caption_info,
        "versions": _versions(),
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    with open(args.output / "run.json", "w") as f:
        json.dump(run, f, indent=2, default=str)
    log.info(
        "Wrote %d objects to %s in %.1fs", export_counts["n_objects"], args.output, timings["total"]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
