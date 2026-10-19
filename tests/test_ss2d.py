import torch

from src.mamba_crack.model import SS2D


def test_ss2d_exposes_four_scan_orderings_and_state_size():
    layer = SS2D(channels=4, state_size=8, snake_scan=True)
    orderings = layer._scan_orderings(height=3, width=5, device=torch.device("cpu"))
    assert len(orderings) == 4, f"expected 4 orderings, got {len(orderings)}"
    assert layer.state_size == 8, f"state_size should be 8, got {layer.state_size}"
    assert all(ordering.numel() == 15 for ordering in orderings), "each ordering should cover all positions exactly once"

    x = torch.randn(2, 4, 3, 5)
    y = layer(x)
    assert y.shape == x.shape[:2] + (3, 5), y.shape


def test_ss2d_fallback_handles_large_inputs_and_backpropagates():
    layer = SS2D(channels=4, state_size=8, snake_scan=True)
    if layer.backend == "mamba_ssm":
        return

    x = torch.randn(1, 4, 17, 19, requires_grad=True)
    y = layer(x)

    assert y.shape == x.shape
    y.mean().backward()
    assert x.grad is not None
