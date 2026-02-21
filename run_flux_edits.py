#!/usr/bin/env python3
"""
Run Flux Kontext on GIE-Bench: load gie_bench.json, for each entry generate an edited
image from the base image using the edit_instruction, save to outputs/{entry_id}.png,
and set entry["edited_image_path"]. Optionally save the modified benchmark JSON.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from diffusers import FluxKontextPipeline

# Flux Kontext config (aligned with EdiVal/baseline_generate/flux.py)
MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
TORCH_DTYPE = torch.bfloat16


def get_device(gpu_id=0):
    if torch.cuda.is_available():
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
    else:
        device = "cpu"
    return device


class FluxKontextGenerator:
    def __init__(self, device="cuda"):
        self.device = device
        print(f"Initializing Flux Kontext on device: {device}")
        self.pipe = FluxKontextPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=TORCH_DTYPE,
        )
        self.pipe.to(device)
        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()
        print("Flux Kontext pipeline initialized.")

    def generate_single_edit(
        self,
        instruction: str,
        current_image: Image.Image,
        guidance_scale: float = 2.5,
        seed: int = None,
    ) -> Image.Image:
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        result = self.pipe(
            image=current_image,
            prompt=instruction,
            guidance_scale=guidance_scale,
        )
        return result.images[0]


def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def run(
    bench_path: str = "gie_bench.json",
    image_dir: str = ".",
    output_dir: str = "outputs",
    output_json_path: Optional[str] = None,
    gpu_id: int = 0,
    seed: int = 1234,
    limit: Optional[int] = None,
):
    bench_path = Path(bench_path)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading benchmark: {bench_path}")
    with open(bench_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected gie_bench.json to be a list of entries")
    entries = data
    if limit is not None:
        entries = entries[:limit]
    total = len(entries)
    print(f"Processing {total} entries.")

    device = get_device(gpu_id)
    generator = FluxKontextGenerator(device=device)

    for entry_id, entry in enumerate(entries):
        image_rel = entry.get("image")
        if not image_rel:
            print(f"[{entry_id}] Skipping: no 'image' key")
            continue
        image_path = image_dir / image_rel.lstrip("/")
        if not image_path.exists():
            print(f"[{entry_id}] Skipping: image not found {image_path}")
            continue
        edit_instruction = entry.get("edit_instruction", "")
        if not edit_instruction:
            print(f"[{entry_id}] Skipping: no 'edit_instruction'")
            continue

        out_path = output_dir / f"{entry_id}.png"
        try:
            current_image = load_image(str(image_path))
            edited = generator.generate_single_edit(
                instruction=edit_instruction,
                current_image=current_image,
                guidance_scale=2.5,
                seed=seed,
            )
            edited.save(out_path)
            # Write path back into the same entry (relative path as in README)
            entry["edited_image_path"] = f"outputs/{entry_id}.png"
            print(f"[{entry_id}] OK -> {entry['edited_image_path']}")
        except Exception as e:
            print(f"[{entry_id}] Error: {e}")
            entry["edited_image_path"] = None

    if output_json_path:
        out_path = Path(output_json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved modified benchmark to {out_path}")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Run Flux Kontext on GIE-Bench; save edited images and optional output JSON."
    )
    parser.add_argument(
        "--bench",
        default="gie_bench.json",
        help="Path to gie_bench.json",
    )
    parser.add_argument(
        "--image-dir",
        default="base_images/",
        help="Base directory for benchmark images (entry['image'] is relative, e.g. /animals/foo.jpg -> image_dir/animals/foo.jpg)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/flux_1_kontext_dev/",
        help="Directory for edited images (default: outputs)",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/flux_1_kontext_dev_results.json",
        help="If set, save modified benchmark with edited_image_path to this path (e.g. results/my_model_output.json)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU id (default: 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for generation",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N entries (for testing)",
    )
    args = parser.parse_args()

    run(
        bench_path=args.bench,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        output_json_path=args.output_json,
        gpu_id=args.gpu,
        seed=args.seed,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
