#!/usr/bin/env python3
"""
Download base images from gie_bench.json URLs into ./base_images.
Uses each entry's 'url' and saves under base_images/ with path from 'image' (e.g. base_images/animals/pexels-mali-75973.jpg).
"""

import argparse
import json
from pathlib import Path

import requests
from tqdm import tqdm


def download_base_images(
    bench_path: str = "gie_bench.json",
    output_dir: str = "base_images",
    timeout: int = 30,
    skip_existing: bool = True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {bench_path}")
    with open(bench_path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected gie_bench.json to be a list of entries")

    for entry in tqdm(data, desc="Downloading"):
        url = entry.get("url")
        image_rel = entry.get("image")
        if not url or not image_rel:
            continue
        # image is e.g. "/animals/pexels-mali-75973.jpg" -> base_images/animals/pexels-mali-75973.jpg
        out_path = output_dir / image_rel.lstrip("/")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if skip_existing and out_path.exists():
            continue
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            out_path.write_bytes(r.content)
        except Exception as e:
            tqdm.write(f"Failed {url[:60]}... -> {out_path}: {e}")

    print(f"Done. Images under {output_dir.absolute()}")


def main():
    parser = argparse.ArgumentParser(description="Download GIE-Bench base images from URLs to ./base_images")
    parser.add_argument("--bench", default="gie_bench.json", help="Path to gie_bench.json")
    parser.add_argument("--output-dir", default="base_images", help="Output directory (default: base_images)")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    parser.add_argument("--no-skip-existing", action="store_true", help="Re-download even if file exists")
    args = parser.parse_args()

    download_base_images(
        bench_path=args.bench,
        output_dir=args.output_dir,
        timeout=args.timeout,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()
