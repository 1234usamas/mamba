from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


FALLBACK_MAX_SEQLEN = 256


class DynamicSnakeConv(nn.Module):
    """Compact learned-offset convolution for curvilinear local features."""

    def __init__(self, in_channels: int, out_channels: int, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled
        self.offset = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1) if enabled else None
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.base_grid = self._make_grid()

    @staticmethod
    def _make_grid() -> torch.Tensor:
        coordinates = torch.tensor(
            [[-1, -1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]],
            dtype=torch.float32,
        )
        return coordinates.view(1, 9, 1, 1, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return F.gelu(self.projection(inputs))
        batch, _, height, width = inputs.shape
        offsets = self.offset(inputs).view(batch, 9, 2, height, width).permute(0, 1, 3, 4, 2)
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=inputs.device),
            torch.linspace(-1, 1, width, device=inputs.device),
            indexing="ij",
        )
        center = torch.stack((x, y), dim=-1).view(1, 1, height, width, 2)
        scale = torch.tensor((2 / max(width - 1, 1), 2 / max(height - 1, 1)), device=inputs.device)
        samples = center + self.base_grid.to(inputs.device) * scale + offsets.tanh() * scale
        sampled = torch.stack(
            [F.grid_sample(inputs, samples[:, index], mode="bilinear", padding_mode="border", align_corners=True) for index in range(9)],
            dim=1,
        ).mean(dim=1)
        return F.gelu(self.projection(sampled))


def diagonal_snake_indices(height: int, width: int, device: torch.device) -> torch.Tensor:
    diagonals: list[list[int]] = []
    for diagonal in range(height + width - 1):
        current = []
        for row in range(height):
            column = diagonal - row
            if 0 <= column < width:
                current.append(row * width + column)
        if diagonal % 2:
            current.reverse()
        diagonals.append(current)
    return torch.tensor([index for diagonal in diagonals for index in diagonal], device=device)


def _invert(indices: torch.Tensor) -> torch.Tensor:
    inverse = torch.empty_like(indices)
    inverse[indices] = torch.arange(indices.numel(), device=indices.device)
    return inverse


def _scan_orders(height: int, width: int, snake: bool, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    base_a = diagonal_snake_indices(height, width, device) if snake else torch.arange(height * width, device=device)
    base_b = torch.arange(height * width, device=device)
    orderings = []
    for base in (base_a, base_b):
        orderings.append((base, _invert(base)))
        orderings.append((base.flip(0), _invert(base.flip(0))))
    return orderings


def _selective_scan_reference(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D_skip: torch.Tensor) -> torch.Tensor:
    batch, length, channels = u.shape
    state_size = A.shape[-1]
    state = u.new_zeros((batch, channels, state_size))
    outputs = []
    for step in range(length):
        delta_t = delta[:, step]
        a_bar = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))
        bu = (delta_t * u[:, step]).unsqueeze(-1) * B[:, step].unsqueeze(1)
        state = a_bar * state + bu
        outputs.append(torch.einsum("bcn,bn->bc", state, C[:, step]))
    y = torch.stack(outputs, dim=1)
    return y + u * D_skip.view(1, 1, -1)


class SS2D(nn.Module):
    """Four-directional 2-D selective state-space scan."""

    def __init__(self, channels: int, state_size: int, snake_scan: bool = True) -> None:
        super().__init__()
        self.snake_scan = snake_scan
        self.channels = channels
        self.state_size = state_size
        self.backend = "fallback_backend"
        self._mamba = None

        try:
            from mamba_ssm import Mamba  # type: ignore

            self._mamba = Mamba(d_model=channels, d_state=state_size, d_conv=3, expand=2)
            self.backend = "mamba_ssm"
        except Exception:
            self._mamba = None
            self.in_norm = nn.LayerNorm(channels)
            self.in_proj = nn.Linear(channels, channels * 2)
            self.dwconv = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels)
            self.dt_proj = nn.Linear(channels, channels)
            self.x_proj = nn.Linear(channels, state_size * 2 + channels)
            self.A_log = nn.Parameter(torch.log(torch.arange(1, state_size + 1, dtype=torch.float32).view(1, state_size).expand(channels, -1).clone()))
            self.skip = nn.Parameter(torch.ones(channels))
            self.output_projection = nn.Linear(channels, channels)

    def _scan_once(self, sequence: torch.Tensor) -> torch.Tensor:
        if self._mamba is not None:
            return self._mamba(sequence)
        projected, gate = self.in_proj(self.in_norm(sequence)).chunk(2, dim=-1)
        x = self.dwconv(projected.transpose(1, 2)).transpose(1, 2)
        x = F.silu(x)
        x_dbl = self.x_proj(x)
        B, C, dt = torch.split(x_dbl, [self.state_size, self.state_size, self.channels], dim=-1)
        delta = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log).to(sequence.device)
        y = _selective_scan_reference(x, delta, A, B, C, self.skip.to(sequence.device))
        y = y * F.silu(gate)
        return self.output_projection(y)

    def _scan_orderings(self, height: int, width: int, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return _scan_orders(height, width, self.snake_scan, device)

    def forward(self, inputs: torch.Tensor, snake_scan: Optional[bool] = None) -> torch.Tensor:
        batch, channels, height, width = inputs.shape
        use_snake = self.snake_scan if snake_scan is None else snake_scan
        scan_inputs = inputs
        if self._mamba is None:
            length = height * width
            if length > FALLBACK_MAX_SEQLEN:
                scale = (FALLBACK_MAX_SEQLEN / length) ** 0.5
                scan_height = max(1, int(height * scale))
                scan_width = max(1, int(width * scale))
                scan_inputs = F.adaptive_avg_pool2d(
                    inputs, (scan_height, scan_width)
                )
        scan_height, scan_width = scan_inputs.shape[-2:]
        sequence = scan_inputs.flatten(2).transpose(1, 2)
        merged = torch.zeros_like(sequence)
        for ordering, inverse in _scan_orders(scan_height, scan_width, use_snake, inputs.device):
            scanned = self._scan_once(sequence.index_select(1, ordering))
            merged = merged + scanned.index_select(1, inverse)
        output = (merged / 4).transpose(1, 2).reshape(batch, channels, scan_height, scan_width)
        if (scan_height, scan_width) != (height, width):
            output = F.interpolate(output, size=(height, width), mode="bilinear", align_corners=False)
        return output


class GatedBottleneckConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 8)
        self.features = nn.Conv2d(channels, hidden * 2, kernel_size=1)
        self.depthwise = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.output = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features, gate = self.features(inputs).chunk(2, dim=1)
        features = F.gelu(self.depthwise(features))
        return inputs + self.output(features * torch.sigmoid(gate))


class MambaSegmenter(nn.Module):
    """Four-stage standalone segmenter: DSConv, SS2D, GBC, and output head."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        model_config = config["model"]
        channels = model_config["base_channels"]
        self.feature_extractor = DynamicSnakeConv(
            model_config["in_channels"], channels, enabled=model_config["use_dsconv"]
        )
        self.mamba_core = SS2D(channels, model_config["state_size"], model_config["snake_scan"])
        self.gated_bottleneck = GatedBottleneckConv(channels)
        self.output_head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, inputs: torch.Tensor, snake_scan: Optional[bool] = None) -> torch.Tensor:
        features = self.feature_extractor(inputs)
        features = features + self.mamba_core(features, snake_scan=snake_scan)
        features = self.gated_bottleneck(features)
        return torch.sigmoid(self.output_head(features))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)