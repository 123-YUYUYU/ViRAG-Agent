import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm.qwen_api_client import QwenAPIClient


def find_sample_image() -> Path:
    image_dir = PROJECT_ROOT / "data" / "page_images"
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        images = sorted(image_dir.glob(pattern))
        if images:
            return images[0]
    raise FileNotFoundError(f"No image found in {image_dir}")


def main() -> None:
    image_path = find_sample_image()
    client = QwenAPIClient()

    start_time = time.perf_counter()
    answer = client.chat(
        question="请简要描述这张图的主要内容。",
        image_paths=[image_path],
    )
    latency = time.perf_counter() - start_time

    print(f"Image: {image_path}")
    print(f"Answer: {answer}")
    print(f"Latency: {latency:.2f} s")


if __name__ == "__main__":
    main()
