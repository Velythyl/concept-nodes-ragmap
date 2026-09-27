"""`ragmap-fetch-weights`: pre-populate the Hugging Face cache with ViT-H-14.

The CLIP ViT-H-14 (laion2b_s32b_b79k, ~3.9 GB) used by algo=CGDetector is not
baked into the image; open_clip loads it from HF_HOME
(/root/.cache/huggingface), which RAGMAP bind-mounts from the host. Run this
once with that mount to download it ahead of a job (or let the first
`ragmap-run` download it).
"""

import sys


def main() -> int:
    import open_clip

    model_name = sys.argv[1] if len(sys.argv) > 1 else "ViT-H-14"
    pretrained = sys.argv[2] if len(sys.argv) > 2 else "laion2b_s32b_b79k"
    open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device="cpu")
    print(f"cached {model_name}/{pretrained}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
