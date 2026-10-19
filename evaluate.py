from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - environment-specific fallback
    plt = None
import numpy as np
import torch
import yaml
from tqdm import tqdm

from mamba_crack.data import make_loaders
from mamba_crack.metrics import batch_metrics
from mamba_crack.model import MambaSegmenter, parameter_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/mamba_segmenter/best.pt"))
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    config["data"]["compute_stats"] = False
    loaders, _, _ = make_loaders(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MambaSegmenter(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    output_dir = args.checkpoint.parent
    rows = []
    inference_images = 0
    inference_seconds = 0.0
    with torch.no_grad():
        for images, masks, names in tqdm(loaders["test"]):
            start = time.perf_counter()
            prediction_scores = model(images.to(device), snake_scan=config["model"]["snake_scan"]).cpu().numpy()[:, 0]
            predictions = prediction_scores > 0.5
            inference_seconds += time.perf_counter() - start
            inference_images += len(images)
            for image, mask, scores, prediction, name in zip(images.numpy(), masks.numpy()[:, 0] > 0.5, prediction_scores, predictions, names):
                row = batch_metrics(scores, mask)
                row["name"] = name
                rows.append(row)
                if plt is None:
                    continue
                figure, axes = plt.subplots(1, 3, figsize=(9, 3))
                axes[0].imshow(image.transpose(1, 2, 0))
                axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
                axes[2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
                figure.tight_layout()
                figure.savefig(output_dir / f"test_{name}.png")
                plt.close(figure)
    with (output_dir / "test_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != "name"}
    report = {
        "metrics": summary,
        "parameters": parameter_count(model),
        "mamba_backend": model.mamba_core.backend,
        "inference_fps": inference_images / max(inference_seconds, 1e-8),
        "inference_device": str(device),
    }
    (output_dir / "test_summary.yaml").write_text(yaml.safe_dump(report), encoding="utf-8")
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    main()