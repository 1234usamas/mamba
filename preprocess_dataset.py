"""Resize, normalize, and rotate paired crack images and segmentation masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


ROTATIONS = (0, 90, 180, 270)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("under-crack-images"),
        help="Directory containing images/ and labels/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("under-crack-images-processed"),
        help="Directory for processed images/ and labels/.",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=(224, 224),
        metavar=("WIDTH", "HEIGHT"),
        help="Output image size in pixels (default: 224 224).",
    )
    parser.add_argument(
        "--mean",
        type=float,
        nargs=3,
        default=(0.485, 0.456, 0.406),
        metavar=("R", "G", "B"),
        help="Per-channel mean after scaling pixels to [0, 1].",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs=3,
        default=(0.229, 0.224, 0.225),
        metavar=("R", "G", "B"),
        help="Per-channel standard deviation after scaling pixels to [0, 1].",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def rotate(image: Image.Image, angle: int) -> Image.Image:
    if angle == 0:
        return image
    return image.rotate(angle, expand=False)


def output_stem(stem: str, angle: int) -> str:
    return stem if angle == 0 else f"{stem}_rot{angle}"


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    size: tuple[int, int],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    overwrite: bool,
) -> int:
    image_dir = input_dir / "images"
    label_dir = input_dir / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError("Input directory must contain images/ and labels/.")

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite to replace it."
            )
        for path in output_dir.rglob("*"):
            if path.is_file():
                path.unlink()
    output_image_dir = output_dir / "images"
    output_label_dir = output_dir / "labels"
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {image_dir}.")

    (output_dir / "normalization.json").write_text(
        json.dumps(
            {
                "scale": "pixel / 255.0",
                "mean": list(mean),
                "std": list(std),
                "apply_to": "RGB image channels",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    processed = 0

    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.png"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {image_path.name}: {label_path}")

        image = Image.open(image_path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
        label = Image.open(label_path).convert("L").resize(size, Image.Resampling.NEAREST)

        for angle in ROTATIONS:
            name = output_stem(image_path.stem, angle)
            rotated_image = rotate(image, angle)
            rotated_label = rotate(label, angle)
            rotated_image.save(output_image_dir / f"{name}.png")
            rotated_label.save(output_label_dir / f"{name}.png")
            processed += 1

    return processed


def main() -> None:
    args = parse_args()
    count = process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        size=(args.size[0], args.size[1]),
        mean=(args.mean[0], args.mean[1], args.mean[2]),
        std=(args.std[0], args.std[1], args.std[2]),
        overwrite=args.overwrite,
    )
    print(f"Processed {count} image-mask pairs into {args.output_dir}")


if __name__ == "__main__":
    main()