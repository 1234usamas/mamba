# Standalone Mamba Crack Segmenter

This repository implements the **Option A component test** from the research brief: a standalone segmenter trained directly on the raw underwater crack images. Results must be described as a standalone segmenter ablation on raw degraded images, not as the final restoration-plus-segmentation system.

## Method

- 70/15/15 image-level split with seed `42`; splitting happens before any training augmentation.
- Every split is resized to `512x512` using bilinear interpolation for RGB images and nearest-neighbor interpolation for binary masks.
- Global RGB mean and standard deviation are computed once from the training split and reused for validation and test.
- Training augmentation is exactly: one random rotation from `0, 90, 180, 270` degrees and a random linear gain from `0.9` to `1.1`. Validation and test receive no augmentation.
- The model is modular: learned-offset DSConv, SS2D scan core, Gated Bottleneck Convolution, and a sigmoid output head.
- The SS2D module reports `mamba_backend`. The current CPU-compatible backend is `pytorch_selective_recurrence`: a state-size-aware diagonal SSM recurrence. It is not a GRU and is not silently presented as the official CUDA selective scan.
- Training loss is `0.5 * softDice + 0.5 * clDice` by default. Betti-error is evaluation-only.

## Setup

Use Python 3.10-3.13 for the official PyTorch and `mamba-ssm` wheels on most CUDA systems. Python 3.14 may not yet have compatible wheels.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The official `mamba-ssm` CUDA extension is optional. On a CUDA development machine with `nvcc`, install it separately with `python -m pip install -r requirements-cuda.txt`. When installed, the model auto-detects the fast backend and logs `backend = "mamba_ssm"`.

SS2D backend policy:
- `mamba_ssm` is the production path for full-scale training.
- The pure-PyTorch fallback is sequential and intentionally slower. To allow full-resolution inputs without an impractical full-resolution recurrence, it adaptively pools SS2D features to at most 256 scan tokens and upsamples the result. This changes the fallback model's scan resolution; use `mamba_ssm` for the full-resolution scan path.
- Any result produced by the fallback must be labeled as a `fallback_backend`, not as `mamba_ssm`.
- The fallback scan is still intended for small/debug runs, not production speed benchmarks.
- The per-stage depth is an explicit implementation choice and should be set deliberately (for example, depth = 1 or 2) rather than inferred implicitly.

The expected dataset layout is:

```text
under-crack-images/
  images/crack0001.jpg
  labels/crack0001.png
```

## Train

```powershell
python train.py --config config.yaml
```

The default configuration uses 512x512 images, batch size 8, AdamW, cosine annealing, up to 150 epochs, early stopping on validation clDice, a 12-epoch raster warmup, and TensorBoard logs. Each epoch prints loss, Dice, clDice, and Betti error for train and validation. These epoch metrics are also saved to `runs/mamba_segmenter/epoch_results.csv` and TensorBoard. The best checkpoint is written to `runs/mamba_segmenter/best.pt`.

### Run on Google Colab GPU

Open [`colab_train.ipynb`](colab_train.ipynb) in Google Colab, select a GPU runtime (**Runtime → Change runtime type → GPU**), and run the notebook cells from top to bottom. It clones this repository, installs the dependencies, extracts the included `under-crack-images.zip`, and trains using the GPU. Mounting Google Drive lets the notebook save checkpoints, epoch results, and evaluation outputs under `MyDrive/mamba_segmenter` so they remain available after the Colab session ends.

The model automatically uses the available Mamba backend. If the optional `mamba-ssm` CUDA extension is unavailable, it uses the documented PyTorch fallback; TensorBoard records the backend used.

Useful ablations:

```powershell
python train.py --no-snake-scan
python train.py --no-dsconv
python train.py --no-cldice
```

## Evaluate

```powershell
python evaluate.py --checkpoint runs/mamba_segmenter/best.pt
```

Evaluation writes per-image overlap metrics, foreground/background class metrics, Betti-error, measured inference FPS, a YAML summary, and a side-by-side test gallery. The headline topology result is `betti_error`; IoU, F1/Dice, precision, recall, and clDice are reported alongside it.

## Notes

The earlier `preprocess_dataset.py` utility creates an offline 224x224 expanded dataset for exploratory use. It is not used by this research protocol, which intentionally applies the exact transforms on the fly after splitting.
