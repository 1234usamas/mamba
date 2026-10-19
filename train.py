from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - environment-specific fallback
    plt = None
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mamba_crack.data import make_loaders
from mamba_crack.losses import TopologyLoss, cldice, soft_dice
from mamba_crack.metrics import batch_metrics
from mamba_crack.model import MambaSegmenter, parameter_count


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_gallery(model, loader, device, path: Path, snake_scan: bool) -> None:
    if plt is None:
        return
    model.eval()
    batch = next(iter(loader))
    images, masks, _ = batch
    with torch.no_grad():
        predictions = model(images[:4].to(device), snake_scan=snake_scan).cpu()
    figure, axes = plt.subplots(len(predictions), 3, figsize=(9, 3 * len(predictions)))
    axes = np.atleast_2d(axes)
    for row, (image, mask, prediction) in enumerate(zip(images[:4], masks[:4], predictions)):
        axes[row, 0].imshow(image.permute(1, 2, 0).numpy() * 0.225 + 0.456)
        axes[row, 1].imshow(mask[0], cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(prediction[0] > 0.5, cmap="gray", vmin=0, vmax=1)
    for axis, title in zip(axes[0], ("Input", "Ground truth", "Prediction")):
        axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def run_epoch(model, loader, criterion, optimizer, device, train: bool, snake_scan: bool) -> dict[str, float]:
    model.train(train)
    totals = {"loss": 0.0, "dice": 0.0, "cldice": 0.0, "count": 0}
    for images, masks, _ in tqdm(loader, leave=False):
        images, masks = images.to(device), masks.to(device)
        with torch.set_grad_enabled(train):
            predictions = model(images, snake_scan=snake_scan)
            loss = criterion(predictions, masks)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch_size = images.shape[0]
        totals["loss"] += loss.item() * batch_size
        totals["dice"] += soft_dice(predictions, masks).item() * batch_size
        totals["cldice"] += cldice(predictions, masks, iterations=10).item() * batch_size
        totals["count"] += batch_size
    return {name: value / totals["count"] for name, value in totals.items() if name != "count"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--no-dsconv", action="store_true")
    parser.add_argument("--no-cldice", action="store_true")
    parser.add_argument("--no-snake-scan", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.image_size is not None:
        config["data"]["image_size"] = args.image_size
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.base_channels is not None:
        config["model"]["base_channels"] = args.base_channels
    if args.output_dir is not None:
        config["training"]["output_dir"] = str(args.output_dir)
    config["model"]["use_dsconv"] = config["model"]["use_dsconv"] and not args.no_dsconv
    config["model"]["use_cldice"] = config["model"]["use_cldice"] and not args.no_cldice
    config["model"]["snake_scan"] = config["model"]["snake_scan"] and not args.no_snake_scan
    seed_everything(config["seed"])
    loaders, _, stats = make_loaders(config)
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output_dir / "normalization.json").write_text(json.dumps({"mean": stats[0], "std": stats[1]}, indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MambaSegmenter(config).to(device)
    criterion = TopologyLoss(config["model"]["use_cldice"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["training"]["epochs"])
    writer = SummaryWriter(output_dir / "tensorboard")
    best_metric = -float("inf")
    stale_epochs = 0
    warmup = config["model"]["snake_scan_warmup_epochs"]
    for epoch in range(config["training"]["epochs"]):
        snake_scan = config["model"]["snake_scan"] and epoch >= warmup
        train_stats = run_epoch(model, loaders["train"], criterion, optimizer, device, True, snake_scan)
        val_stats = run_epoch(model, loaders["val"], criterion, optimizer, device, False, snake_scan)
        scheduler.step()
        epoch_metrics = {f"train/{key}": value for key, value in train_stats.items()}
        epoch_metrics.update({f"val/{key}": value for key, value in val_stats.items()})
        for name, value in epoch_metrics.items():
            writer.add_scalar(name, value, epoch)
        writer.add_scalar("scan/snake_enabled", int(snake_scan), epoch)
        if val_stats["cldice"] > best_metric:
            best_metric = val_stats["cldice"]
            stale_epochs = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, output_dir / "best.pt")
        else:
            stale_epochs += 1
        if (epoch + 1) % config["training"]["gallery_every"] == 0:
            save_gallery(model, loaders["val"], device, output_dir / f"val_epoch_{epoch + 1:03d}.png", snake_scan)
        print(f"epoch={epoch + 1} snake={snake_scan} train={train_stats} val={val_stats}")
        if stale_epochs >= config["training"]["patience"]:
            break
    writer.add_text("model", f"parameters={parameter_count(model)} backend={model.mamba_core.backend}")
    writer.close()


if __name__ == "__main__":
    main()