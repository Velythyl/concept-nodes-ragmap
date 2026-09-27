# concept-nodes (ConceptGraphs, algo=CGDetector) as a RAGMAP object_mapping stage.
#
# torch 2.4.1 from PyPI is the CUDA 12.1 build and ships its own CUDA runtime
# libraries (nvidia-* wheels), so the lean `base` flavour of the matching CUDA
# image is enough: it only has to carry the env the NVIDIA container toolkit /
# CDI keys off. GPUs are injected at run time with
# `podman run --device nvidia.com/gpu=<uuid|all>`.
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

ARG UV_VERSION=0.9.18
ARG GIT_SHA=unknown
ENV DEBIAN_FRONTEND=noninteractive \
    CONCEPT_NODES_ROOT=/opt/concept-nodes \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/root/.local/bin:$PATH \
    PYTHONPATH=/opt/concept-nodes \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    YOLO_CONFIG_DIR=/opt/ultralytics \
    CONCEPT_NODES_WEIGHTS=/opt/weights \
    UV_NO_CACHE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git \
        libgl1 libegl1 libglib2.0-0 libgomp1 libx11-6 libxext6 libxrender1 libsm6 \
    && rm -rf /var/lib/apt/lists/* \
    && curl --fail --location --silent --show-error https://astral.sh/uv/${UV_VERSION}/install.sh | sh \
    && uv python install 3.10 \
    && uv venv --python 3.10 ${VIRTUAL_ENV}

COPY docker/requirements.txt /tmp/requirements.txt
RUN uv pip install --python ${VIRTUAL_ENV}/bin/python -r /tmp/requirements.txt \
    && python -c "import torch; assert torch.version.cuda == '12.1', torch.version.cuda"

# Small checkpoints are baked in (checksummed). CLIP ViT-H-14 (~3.9 GB) is
# not: it is read from the mounted HF cache -- see README-ragmap.md.
RUN mkdir -p /opt/weights ${YOLO_CONFIG_DIR} \
    && cd /opt/weights \
    && curl -fsSL -O https://github.com/ultralytics/assets/releases/download/v8.2.0/mobile_sam.pt \
    && curl -fsSL -O https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-world.pt \
    && curl -fsSL -O https://raw.githubusercontent.com/concept-graphs/concept-graphs/66175d63f466d264edce9f1fb6987c5ba1dcac0e/conceptgraph/scannet200_classes.txt \
    && printf '%s  %s\n' \
        6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f mobile_sam.pt \
        095f5266bb9b654bd5ad9e21e9cdeda78e0f2c8460f5d652eaf04bab7ee251cf yolov8s-world.pt \
        4e4c4ef1e39dad24538b8feeba5be43abca39b21c5e5492ca6413f2908ec49ad scannet200_classes.txt \
        | sha256sum -c -

COPY . ${CONCEPT_NODES_ROOT}

# Instantiating the detector once bakes YOLO-World's CLIP ViT-B/32 text
# encoder into /root/.cache/clip (outside the mounted HF cache) and proves the
# imports work without a GPU.
RUN cd /tmp && python -c "\
import open3d, open_clip, mobile_sam; \
from concept_graphs.perception.segmentation.YoloMobileSAM import YoloMobileSAM; \
YoloMobileSAM('/opt/weights/yolov8s-world.pt', '/opt/weights/scannet200_classes.txt', 'cpu', 'vit_t', '/opt/weights/mobile_sam.pt', 'bbox', 'cpu', False)" \
    && ls -la /root/.cache/clip \
    && for tool in run:ragmap-run smoke:ragmap-smoke fetch_weights:ragmap-fetch-weights; do \
        printf '#!/bin/sh\nexec /opt/venv/bin/python -m ragmap_bridge.%s "$@"\n' "${tool%%:*}" > /usr/local/bin/${tool##*:}; \
        chmod +x /usr/local/bin/${tool##*:}; \
    done

ENV CONCEPT_NODES_GIT_SHA=${GIT_SHA}
WORKDIR /output
ENTRYPOINT ["ragmap-run"]
CMD ["--help"]
