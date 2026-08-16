"""Pinned, deterministic P2P metric implementations."""

from __future__ import annotations

import hashlib
import os
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


DINO_SOURCE = Path(os.environ.get("FDCA_DINOV2_SOURCE", "external/metric_sources/dinov2"))
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def tensor_sha256(x: torch.Tensor) -> str:
    a = x.detach().cpu().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


def preprocessing_contract() -> dict[str, Any]:
    return {
        "decoded": "RGB float32 [-1,1], NCHW, native 256x256",
        "lpips": "identity; official LPIPS net=alex version=0.1 normalize=False",
        "dinov2": "map [-1,1] to [0,1]; bicubic 224x224 antialias=True; ImageNet mean/std",
        "classifier": "map [-1,1] to [0,1]; bilinear resize shorter side 232 antialias=True; center crop 224; ImageNet mean/std",
        "ssim": "11x11 Gaussian sigma=1.5 reflect padding; RGB-channel and pixel mean; data_range=2; K1=.01 K2=.03",
        "pixel_l1": "mean absolute difference in [-1,1]",
        "psnr": "10 log10(4/MSE) in [-1,1]; +inf iff exact",
    }


def _mean_std(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype)[None, :, None, None]
    std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype)[None, :, None, None]
    return mean, std


def dino_preprocess(x: torch.Tensor) -> torch.Tensor:
    x01 = (x + 1.0) * 0.5
    resized = F.interpolate(x01, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
    mean, std = _mean_std(resized.device, resized.dtype)
    return (resized - mean) / std


def classifier_preprocess(x: torch.Tensor) -> torch.Tensor:
    x01 = (x + 1.0) * 0.5
    h, w = x01.shape[-2:]
    if h <= w:
        nh, nw = 232, int(round(w * 232 / h))
    else:
        nh, nw = int(round(h * 232 / w)), 232
    resized = F.interpolate(x01, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)
    top = (nh - 224) // 2
    left = (nw - 224) // 2
    cropped = resized[:, :, top:top + 224, left:left + 224]
    mean, std = _mean_std(cropped.device, cropped.dtype)
    return (cropped - mean) / std


def load_lpips(device: str = "cuda") -> torch.nn.Module:
    import lpips
    return lpips.LPIPS(net="alex", version="0.1", pretrained=True, eval_mode=True, verbose=False).float().eval().to(device)


def load_dino(device: str = "cuda") -> torch.nn.Module:
    model = torch.hub.load(str(DINO_SOURCE), "dinov2_vits14", source="local", pretrained=True)
    return model.float().eval().to(device)


def load_classifier(device: str = "cuda") -> torch.nn.Module:
    from torchvision.models import ResNet50_Weights, resnet50
    return resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).float().eval().to(device)


@torch.inference_mode()
def lpips_distance(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return model(x, y, normalize=False).reshape(-1).float()


@torch.inference_mode()
def dino_features(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    return F.normalize(model(dino_preprocess(x)).float(), dim=1)


@torch.inference_mode()
def dino_distance(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    result = 1.0 - (dino_features(model, x) * dino_features(model, y)).sum(dim=1)
    identical = (x == y).reshape(len(x), -1).all(dim=1)
    return torch.where(identical, torch.zeros_like(result), result)


@torch.inference_mode()
def classifier_outputs(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, class_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    lx = model(classifier_preprocess(x)).float()
    ly = model(classifier_preprocess(y)).float()
    px = torch.softmax(lx, dim=1)
    py = torch.softmax(ly, dim=1)
    m = 0.5 * (px + py)
    eps = torch.finfo(px.dtype).tiny
    jsd = 0.5 * ((px * (px.clamp_min(eps).log() - m.clamp_min(eps).log())).sum(1) + (py * (py.clamp_min(eps).log() - m.clamp_min(eps).log())).sum(1))
    idx = class_ids.to(device=lx.device, dtype=torch.long).reshape(-1, 1)
    pxc = px.gather(1, idx)[:, 0]
    pyc = py.gather(1, idx)[:, 0]
    lxc = lx.gather(1, idx)[:, 0]
    lyc = ly.gather(1, idx)[:, 0]
    return {
        "classifier_jsd": jsd,
        "classifier_top1_disagreement": (px.argmax(1) != py.argmax(1)).to(torch.int64),
        "condition_probability_abs_change": (pxc - pyc).abs(),
        "condition_probability_drop": pxc - pyc,
        "condition_logit_abs_change": (lxc - lyc).abs(),
        "reference_top1": px.argmax(1),
        "alternative_top1": py.argmax(1),
        "reference_condition_probability": pxc,
        "alternative_condition_probability": pyc,
        "reference_condition_logit": lxc,
        "alternative_condition_logit": lyc,
    }


def _gaussian_kernel(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coord = torch.arange(11, device=device, dtype=dtype) - 5
    g = torch.exp(-(coord * coord) / (2 * 1.5 * 1.5))
    g = g / g.sum()
    k = torch.outer(g, g)
    return k[None, None].repeat(3, 1, 1, 1)


def pixel_metrics(x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
    diff = x - y
    l1 = diff.abs().mean(dim=(1, 2, 3))
    mse = (diff * diff).mean(dim=(1, 2, 3))
    psnr = torch.where(mse == 0, torch.full_like(mse, torch.inf), 10.0 * torch.log10(4.0 / mse))
    k = _gaussian_kernel(x.device, x.dtype)
    xp = F.pad(x, (5, 5, 5, 5), mode="reflect")
    yp = F.pad(y, (5, 5, 5, 5), mode="reflect")
    mux = F.conv2d(xp, k, groups=3)
    muy = F.conv2d(yp, k, groups=3)
    mux2, muy2, muxy = mux * mux, muy * muy, mux * muy
    sigx = F.conv2d(xp * xp, k, groups=3) - mux2
    sigy = F.conv2d(yp * yp, k, groups=3) - muy2
    sigxy = F.conv2d(xp * yp, k, groups=3) - muxy
    c1, c2 = (0.01 * 2.0) ** 2, (0.03 * 2.0) ** 2
    ssim_map = ((2 * muxy + c1) * (2 * sigxy + c2)) / ((mux2 + muy2 + c1) * (sigx + sigy + c2))
    return {"ssim": ssim_map.mean(dim=(1, 2, 3)), "pixel_l1": l1, "psnr": psnr}


def preprocessing_hashes(sample: torch.Tensor) -> dict[str, str]:
    return {
        "contract_sha256": hashlib.sha256(json.dumps(preprocessing_contract(), sort_keys=True).encode()).hexdigest(),
        "decoded_sample_sha256": tensor_sha256(sample),
        "dino_preprocessed_sha256": tensor_sha256(dino_preprocess(sample)),
        "classifier_preprocessed_sha256": tensor_sha256(classifier_preprocess(sample)),
    }
