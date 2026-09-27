"""RAGMAP scene export -> rgbd_dataset.

Layout (the RAGMAP object_mapping contract)::

    <base_path>/meta.json      {"fx","fy","cx","cy","width","height",
                                "depth_scale", "up_axis", "frame"}
    <base_path>/frames.jsonl   one {"frame_index", "observation_id", "rgb",
                                "depth", "pose"} per line; pose is 16 floats,
                                row-major 4x4 camera-to-world, OpenCV camera
                                convention (x right, y down, z forward).

`rgb` and `depth` are relative to `base_path`. Depth is a uint16 PNG in
`depth_scale` units per metre. Intrinsics are global (from meta.json).

Lives in concept-nodes rather than in the `rgbd_dataset` submodule so the fork
does not need to carry a submodule patch. It is registered as the hydra dataset
config `dataset=RagmapScene` (conf/dataset/RagmapScene.yaml).
"""

import json
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from rgbd_dataset.rgbd_dataset.BaseRGBDDataset import BaseRGBDDataset


def read_meta(base_path: Union[str, Path]) -> dict:
    with open(Path(base_path) / "meta.json", "r") as f:
        return json.load(f)


def read_frames(base_path: Union[str, Path]) -> List[dict]:
    frames = []
    with open(Path(base_path) / "frames.jsonl", "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for key in ("frame_index", "rgb", "depth", "pose"):
                if key not in record:
                    raise ValueError(f"frames.jsonl line {line_no} is missing {key!r}")
            if len(record["pose"]) != 16:
                raise ValueError(
                    f"frames.jsonl line {line_no}: pose must be 16 floats, got {len(record['pose'])}"
                )
            frames.append(record)
    if not frames:
        raise ValueError(f"{Path(base_path) / 'frames.jsonl'} holds no frames")
    frames.sort(key=lambda r: int(r["frame_index"]))
    return frames


class RagmapScene(BaseRGBDDataset):
    def __init__(
        self,
        base_path: Union[str, Path],
        dataset_name: str = "ragmap",
        scene: str = "scene",
        width: Optional[int] = None,
        height: Optional[int] = None,
        depth_scale: Optional[float] = None,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        **kwargs,
    ):
        # Everything the contract carries comes from meta.json; explicit
        # arguments (hydra overrides) win so a caller can still correct a value.
        self.root = Path(base_path)
        self.meta = read_meta(self.root)
        self.frames = read_frames(self.root)

        def pick(value, key):
            return value if value is not None else self.meta[key]

        super().__init__(
            dataset_name=dataset_name,
            base_path=self.root,
            scene=scene,
            width=int(pick(width, "width")),
            height=int(pick(height, "height")),
            depth_scale=float(pick(depth_scale, "depth_scale")),
            fx=float(pick(fx, "fx")),
            fy=float(pick(fy, "fy")),
            cx=float(pick(cx, "cx")),
            cy=float(pick(cy, "cy")),
            **kwargs,
        )
        # Kept in the same sliced order as the paths/poses in BaseRGBDDataset.
        self.frame_indices = [int(r["frame_index"]) for r in self.frames][
            self.sequence_start : self.sequence_end : self.sequence_stride
        ]
        self.observation_ids = [r.get("observation_id") for r in self.frames][
            self.sequence_start : self.sequence_end : self.sequence_stride
        ]

    def get_rgb_paths(self) -> List[str]:
        return [str(self.root / r["rgb"]) for r in self.frames]

    def get_depth_paths(self) -> List[str]:
        return [str(self.root / r["depth"]) for r in self.frames]

    def get_se3_poses(self) -> List[np.ndarray]:
        return [
            np.asarray(r["pose"], dtype=np.float64).reshape((4, 4)) for r in self.frames
        ]

    def read_rgb(self, path: Union[str, Path]) -> np.ndarray:
        if not Path(path).is_file():
            raise FileNotFoundError(f"RGB frame not found: {path}")
        return super().read_rgb(path)

    def read_depth(self, path: Union[str, Path]) -> np.ndarray:
        depth = super().read_depth(path)
        if depth is None:
            raise FileNotFoundError(f"Depth frame not found or unreadable: {path}")
        return depth

    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        result["frame_index"] = self.frame_indices[idx]
        return result
