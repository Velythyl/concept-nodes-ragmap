# concept-nodes as a RAGMAP `object_mapping` stage

Image: `ghcr.io/velythyl/concept-nodes-ragmap` (built by
`.github/workflows/ghcr.yml` on every push to `main`; tags `latest`, `main`,
`sha-<7>`). Pin `sha-<7>` in RAGMAP.

## Run

```bash
podman run --rm --arch amd64 \
  --device nvidia.com/gpu=<GPU-UUID> -e CUDA_VISIBLE_DEVICES=0 \
  --network host \
  -e RAGMAP_VLM_BASE_URL=http://127.0.0.1:<port> -e RAGMAP_VLM_MODEL=<profile> \
  -v <scene_export>:/input:ro,Z \
  -v <object_map_out>:/output:Z \
  -v <hf_cache>:/root/.cache/huggingface:Z \
  ghcr.io/velythyl/concept-nodes-ragmap:sha-<7> \
  --input /input --output /output --caption
```

* GPU args: exactly what `ragmap.containers.gpu_device_args()` returns (after
  `isolate_cdi_spec()`), as for vggt_slam2.
* `--network host` and the two `RAGMAP_VLM_*` variables are only needed with
  `--caption`. `ragmap-vlm-service` binds `127.0.0.1` by default, so the
  container must share the host network namespace to reach it (podman `-p`
  publishing is not involved).
* `RAGMAP_VLM_MODEL` must be the service's **profile name** (e.g.
  `qwen3_vl_8b`); the service 400s any other `model`. If unset, `ragmap-run`
  reads it from `GET /status` (ragmap-vlm-service) or `GET /v1/models` (vLLM).
* Without `--caption` nothing leaves the container; labels are the
  YOLO-World class (majority vote over the object's detections).
* Extra args after the flags are hydra overrides for concept-nodes, e.g.
  `sim_thresh=0.89`. Other flags: `--stride N`, `--max-frames N`,
  `--device cpu`, `--algo CG`.

## Weights

| weight | where |
| --- | --- |
| `mobile_sam.pt`, `yolov8s-world.pt`, `scannet200_classes.txt` | baked, `/opt/weights` (sha256-checked) |
| CLIP ViT-B/32 (YOLO-World text encoder) | baked, `/root/.cache/clip` |
| CLIP ViT-H-14 `laion2b_s32b_b79k` (~3.9 GB, object features) | **not baked**: HF cache, `hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K` under `HF_HOME=/root/.cache/huggingface` |

Mount the host HF cache at `/root/.cache/huggingface` (the same mount
vggt_slam2 uses). If the model is not there yet the first run downloads it;
to pre-fetch: `podman run --rm -v <hf_cache>:/root/.cache/huggingface:Z
--entrypoint ragmap-fetch-weights ghcr.io/velythyl/concept-nodes-ragmap:latest`.

## Contract

Input `/input/meta.json` + `/input/frames.jsonl` (poses camera-to-world,
OpenCV camera, row-major), read by `concept_graphs/datasets/RagmapScene.py`
(hydra `dataset=RagmapScene`). Output `/output/objects.jsonl` (one object per
line, contract fields; extensions only under `extra`), `objects/<id>.ply`,
`crops/<id>.png`, `clip_features.npy` (row `extra.clip_feature_row`), and
`run.json` (resolved config, counts, timings, versions, VLM endpoint).
All coordinates are in the input world frame.

`extra` keys: `label_source` (`vlm_tag` | `detector` | `fallback`),
`detector_label`, `detector_label_votes`, `vlm_tag`, `vlm_caption_raw`,
`vlm_background` (the VLM answered BACKGROUND; object is kept), `n_detections`,
`n_points`, `clip_feature_row`, `crop_frame_index`.

## Captioner request shape

`concept_graphs/vlm/OpenAICaptioner.py` posts to `<base>/v1/chat/completions`:
one `user` message (system prompt folded into the text; the RAGMAP service
rejects `system`), text first then up to 4 `image_url` blocks as
`data:image/jpeg;base64,...`, `temperature: 0.0`, `max_tokens: 64`, `model` =
profile. `data:` URLs are used rather than `file://` because container paths
are not the service's paths. A failed request yields caption `Invalid`
(reported as `null`); `run.json.caption` counts failures and the run exits
non-zero if every request failed.

## Smoke test

`podman run --rm --entrypoint ragmap-smoke <image> --device cpu` — 5 synthetic
frames (ultralytics' `bus.jpg` over a flat 4 m depth), a fake local VLM that
enforces ragmap-vlm-service's request validation, and contract checks
(including the world-frame z). Runs in CI after every publish. It uses
ViT-B-32 CLIP (`ft_extraction=CLIP`) to keep the download small.
