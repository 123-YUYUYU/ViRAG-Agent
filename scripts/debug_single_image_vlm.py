import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PAGE_IMAGES_DIR, QWEN_VL_MODEL_NAME  # noqa: E402
from llm.qwen2_vl_client import Qwen2VLClient  # noqa: E402


DEFAULT_QUESTION = "请简要描述这张图的主要内容。"


def cuda_memory_snapshot() -> str:
    if not torch.cuda.is_available():
        return "cuda_available=False"
    return (
        f"allocated={torch.cuda.memory_allocated()} "
        f"reserved={torch.cuda.memory_reserved()}"
    )


def find_default_image() -> Path:
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        matches = sorted(Path(PAGE_IMAGES_DIR).rglob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No image found under {PAGE_IMAGES_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Qwen2-VL on one page image without retriever/reranker/agent."
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to one image under data/page_images. Defaults to the first image found.",
    )
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--model", default=str(QWEN_VL_MODEL_NAME))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image) if args.image else find_default_image()
    image_path = image_path.resolve()

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    with Image.open(image_path) as image:
        print(
            f"[SINGLE_IMAGE_DEBUG] image_path={image_path} "
            f"width={image.width} height={image.height} mode={image.mode}"
        )

    print(f"[SINGLE_IMAGE_DEBUG] before client init cuda_memory {cuda_memory_snapshot()}")
    client = Qwen2VLClient(
        model_name=args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"[SINGLE_IMAGE_DEBUG] after client init cuda_memory {cuda_memory_snapshot()}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": args.question},
            ],
        }
    ]
    print(f"[SINGLE_IMAGE_DEBUG] question={args.question}")
    answer = client.chat(messages, do_sample=False)
    print("[SINGLE_IMAGE_DEBUG] answer:")
    print(answer)
    print(f"[SINGLE_IMAGE_DEBUG] final cuda_memory {cuda_memory_snapshot()}")


if __name__ == "__main__":
    main()
