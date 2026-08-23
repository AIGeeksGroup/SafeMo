from __future__ import annotations

from typing import Dict

import torch


def to_btd(value: torch.Tensor) -> torch.Tensor:
    return value.transpose(1, 3).squeeze(2)


def masked_vector_l2(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    error = (to_btd(prediction) - to_btd(target)).pow(2).sum(dim=-1).sqrt()
    return (error * mask).sum() / mask.sum().clamp_min(1.0)


def _derivative(value: torch.Tensor, order: int) -> torch.Tensor:
    result = to_btd(value)
    for _ in range(order):
        result = result[:, 1:] - result[:, :-1]
    return result


def _derivative_mask(mask: torch.Tensor, order: int) -> torch.Tensor:
    result = mask
    for _ in range(order):
        result = result[:, 1:] * result[:, :-1]
    return result


def derivative_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    order: int,
    frequency_mode: str = "none",
    frequency_weight: float = 0.0,
) -> torch.Tensor:
    residual = _derivative(prediction, order) - _derivative(target, order)
    valid = _derivative_mask(mask, order)
    temporal = (residual.pow(2).sum(dim=-1).sqrt() * valid).sum()
    temporal = temporal / valid.sum().clamp_min(1.0)
    if frequency_mode == "none" or frequency_weight == 0.0:
        return temporal
    spectrum = torch.fft.rfft(residual.float(), dim=1).abs()
    frequencies = torch.linspace(
        0.0, 1.0, spectrum.shape[1], device=spectrum.device, dtype=spectrum.dtype
    ).view(1, -1, 1)
    if frequency_mode == "linear":
        weights = frequencies
    elif frequency_mode == "log":
        weights = torch.log1p(9.0 * frequencies)
    else:
        raise ValueError("Unknown frequency mode: {}".format(frequency_mode))
    spectral = (spectrum * weights).mean()
    return temporal + float(frequency_weight) * spectral.to(temporal.dtype)


def kinematic_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    frequency_mode: str = "none",
    frequency_weight: float = 0.0,
) -> Dict[str, torch.Tensor]:
    return {
        "pose": masked_vector_l2(prediction, target, mask),
        "velocity": derivative_loss(
            prediction, target, mask, 1, frequency_mode, frequency_weight
        ),
        "acceleration": derivative_loss(
            prediction, target, mask, 2, frequency_mode, frequency_weight
        ),
    }


def segment_shuffle(
    motion: torch.Tensor, lengths: torch.Tensor, segments: int = 4
) -> torch.Tensor:
    output = motion.clone()
    for index in range(motion.shape[0]):
        length = int(lengths[index].item())
        if length < segments:
            continue
        boundaries = torch.linspace(0, length, segments + 1, dtype=torch.long)
        pieces = [motion[index : index + 1, ..., boundaries[i] : boundaries[i + 1]] for i in range(segments)]
        order = torch.randperm(segments).tolist()
        output[index : index + 1, ..., :length] = torch.cat(
            [pieces[i] for i in order], dim=-1
        )
    return output


def time_reverse(motion: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    output = motion.clone()
    for index in range(motion.shape[0]):
        length = int(lengths[index].item())
        output[index : index + 1, ..., :length] = torch.flip(
            motion[index : index + 1, ..., :length], dims=(-1,)
        )
    return output


def decouple_motion(
    motion: torch.Tensor, lengths: torch.Tensor, segments: int = 4
) -> torch.Tensor:
    if torch.rand((), device=motion.device).item() < 0.5:
        return segment_shuffle(motion, lengths, segments)
    return time_reverse(motion, lengths)


def masked_pool(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    sequence = to_btd(value)
    return (sequence * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
        dim=1, keepdim=True
    ).clamp_min(1.0)
