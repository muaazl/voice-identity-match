"""
Model acquisition script for VoiceMimic ONNX assets.
Downloads DTLN (Speech Enhancement), Silero VAD (Voice Activity Detection),
and CAM++ (192-D Speaker Verification) ONNX models into onnx_models/.
"""

import os
import sys
import logging
import requests
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS = {
    "dtln_model_1.onnx": [
        "https://huggingface.co/niobures/DTLN/resolve/main/models/DTLN/onnx/model_1.onnx",
    ],
    "dtln_model_2.onnx": [
        "https://huggingface.co/niobures/DTLN/resolve/main/models/DTLN/onnx/model_2.onnx",
    ],
    "silero_vad.onnx": [
        "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx",
        "https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx",
    ],
    "campplus.onnx": [
        "https://huggingface.co/welcomyou/campplus-3dspeaker-200k-onnx/resolve/main/campplus_cn_en_common_200k.onnx",
    ],
}


def download_file(urls: list[str], destination: Path, chunk_size: int = 1024 * 64) -> bool:
    """Download a file from a list of candidate URLs with fallback and progress bar."""
    if destination.exists() and destination.stat().st_size > 10000:
        logger.info(f"Asset already present: {destination.name} ({destination.stat().st_size / 1024 / 1024:.2f} MB)")
        return True

    temp_path = destination.with_suffix(".tmp")
    headers = {"User-Agent": "VoiceMimic-ModelDownloader/1.0"}

    for url in urls:
        logger.info(f"Downloading {destination.name} from {url}...")
        try:
            with requests.get(url, stream=True, timeout=60, headers=headers) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))

                with open(temp_path, "wb") as f, tqdm(
                    desc=destination.name,
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))

            # Atomic move
            temp_path.replace(destination)
            logger.info(f"Successfully saved {destination.name} ({destination.stat().st_size / 1024 / 1024:.2f} MB)")
            return True
        except Exception as e:
            logger.warning(f"Failed to download from {url}: {e}")
            if temp_path.exists():
                temp_path.unlink()

    logger.error(f"All candidate URLs exhausted for {destination.name}")
    return False


def main(target_dir: str = "onnx_models"):
    dest_path = Path(target_dir).resolve()
    dest_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Target model directory: {dest_path}")

    all_success = True
    for filename, urls in MODELS.items():
        filepath = dest_path / filename
        success = download_file(urls, filepath)
        if not success:
            all_success = False

    if not all_success:
        logger.error("One or more models failed to download.")
        sys.exit(1)

    logger.info("All ONNX models downloaded and verified.")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "onnx_models"
    main(target)
