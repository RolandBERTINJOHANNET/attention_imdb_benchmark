#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
function_eval_attgw.py
----------------------
Importable, CLI-free evaluator for a trained GlobalWorkspaceFusion model.
Runs the **CLEAN** evaluation (normalize="none") on MM-IMDb 1.0 aug-70,
using image_latents + caption_embeddings to predict labels, and returns metrics.

Usage (from another script):
    from function_eval_attgw import evaluate_clean
    metrics = evaluate_clean(model, backbone="blip2", device="cuda")  # or backbone="clip"

Optional:
    # Evaluate with random attention at test time (spiky with small temperature)
    metrics = evaluate_clean(model, backbone="clip",
                             selector_override="random", temperature=0.05)

Returns a dict with:
    {
      "status": "OK",
      "img_dim": int, "txt_dim": int, "lbl_dim": int,
      "bce_val": float, "bce_test": float,
      "macro_f1_val": float, "macro_f1_test": float,
      "n_val": int, "n_test": int,
      "batch_size": int, "threshold": float,
      "view": "clean", "normalize": "none",
      "backbone": "clip" | "blip2"
    }
"""

from __future__ import annotations
from typing import Dict, Tuple, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score

# ==== Your codebase bits (same as training) ===================================
from imdb_gw.overrides import shimmer_patches as _sp  # apply patches
from shimmer.modules.global_workspace import GlobalWorkspaceFusion
# IMPORTANT: import BOTH builders and select by backbone
from imdb_gw.data.dataset_1x_aug70 import (
    make_datamodule_blip2_aug70,
    make_datamodule_clip_aug70,
)

# ==== Fixed eval config (mirrors clean-eval) ==================================
VIEW_MODE_EVAL  = "clean"
BATCH_SIZE_DEF  = 1024
THRESHOLD_DEF   = 0.50

# No extra normalization: extractor already did dataset-wise z-scoring.
NORMALIZE_FLAG  = "none"


# =============================================================================
# RandomSelection (softmax of random scores / temperature) -- matches training
# =============================================================================
class RandomSelection(nn.Module):
    """
    Uniform-softmax attention across present domains, controlled by 'temperature'.
    Lower temperature -> peakier (more extreme) weights.
    Returns per-domain coefficients of shape [B].
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, domains: Dict[str, torch.Tensor], encodings_pre_fusion: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not isinstance(domains, dict) or len(domains) == 0:
            raise ValueError("RandomSelection expects a non-empty dict of domain tensors.")
        first = next(iter(domains.values()))
        if not torch.is_tensor(first):
            raise ValueError("RandomSelection expects domain tensors as values.")
        device = first.device
        batch_size = first.shape[0]
        names = list(domains.keys())
        num_domains = len(names)
        scores = torch.rand(batch_size, num_domains, device=device)
        scores = torch.softmax(scores / max(self.temperature, 1e-8), dim=1)
        return {name: scores[:, i] for i, name in enumerate(names)}


# =============================================================================
# Datamodule factory (CHOOSES FEATURES BY BACKBONE)
# =============================================================================
def _build_dm_clean_for(backbone: str, batch_size: int):
    b = backbone.lower()
    if b == "blip2":
        return make_datamodule_blip2_aug70(
            batch_size=batch_size, num_workers=8, pin_memory=True,
            normalize=NORMALIZE_FLAG, view_mode=VIEW_MODE_EVAL, include_labels=True
        )
    elif b == "clip":
        return make_datamodule_clip_aug70(
            batch_size=batch_size, num_workers=8, pin_memory=True,
            normalize=NORMALIZE_FLAG, view_mode=VIEW_MODE_EVAL, include_labels=True
        )
    else:
        raise ValueError(f"Unknown backbone '{backbone}' (expected 'clip' or 'blip2').")


# =============================================================================
# Data helpers (CLEAN view)
# =============================================================================
def _get_dims_from_dm(dm) -> Tuple[int, int, int]:
    key3 = frozenset(["image_latents","caption_embeddings","labels"])
    tr = dm.train_datasets[key3]
    img_dim = int(tr.domain_data["image_latents"].shape[1])
    txt_dim = int(tr.domain_data["caption_embeddings"].shape[1])
    lbl_dim = int(tr.domain_data["labels"].shape[1])
    return img_dim, txt_dim, lbl_dim

def _build_clean_loaders(dm, device: torch.device, batch_size: int):
    key3 = frozenset(["image_latents","caption_embeddings","labels"])
    tr = dm.train_datasets[key3]
    va = dm.val_datasets[key3]
    te = dm.test_datasets.get(key3, va)

    def mk(ds):
        xi = ds.domain_data["image_latents"].contiguous()
        xt = ds.domain_data["caption_embeddings"].contiguous()
        y  = ds.domain_data["labels"].contiguous()
        return DataLoader(TensorDataset(xi, xt, y),
                          batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=(device.type == "cuda"))
    return mk(tr), mk(va), mk(te)


# =============================================================================
# Inference (same grouped flow as reference eval)
# =============================================================================
@torch.no_grad()
def _infer_logits(model: GlobalWorkspaceFusion, loader: DataLoader, device: torch.device) -> np.ndarray:
    use_amp = (device.type == "cuda")
    dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    TRI = frozenset(["image_latents","caption_embeddings","labels"])
    outs: List[np.ndarray] = []

    for xi, xt, _ in iter(loader):
        xi = xi.to(device, non_blocking=True)
        xt = xt.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype):
            grouped_in = {TRI: {"image_latents": xi, "caption_embeddings": xt}}
            enc_all    = model.encode_domains(grouped_in)
            fused_all  = model.encode_and_fuse({TRI: enc_all[TRI]}, model.selection_mod)
            z          = fused_all[TRI]
            dec_all    = model.decode({TRI: z}, domains=["labels"])
            logits     = dec_all[TRI]["labels"]

        outs.append(logits.float().cpu().numpy())

    return np.concatenate(outs, 0) if outs else np.zeros((0, 0), np.float32)


def _collect_labels(loader: DataLoader) -> np.ndarray:
    ys = [yb.numpy() for _,_,yb in loader]
    return np.concatenate(ys, 0).astype(np.int64) if ys else np.zeros((0,0), np.int64)


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# =============================================================================
# Public API
# =============================================================================
def evaluate_clean(
    model: GlobalWorkspaceFusion,
    backbone: str,
    device: Optional[Union[str, torch.device]] = None,
    batch_size: int = BATCH_SIZE_DEF,
    threshold: float = THRESHOLD_DEF,
    *,
    selector_override: Optional[str] = None,   # None | "shared" | "random"
    temperature: float = .1,                  # used if selector_override == "random"
) -> Dict:
    """
    Evaluate a trained GlobalWorkspaceFusion model on the CLEAN split.

    Args:
        model:       Trained GW model (already configured as trained).
        backbone:    "clip" or "blip2" — selects the correct feature datamodule for eval.
        device:      "cuda", "cpu", or torch.device. Defaults to CUDA if available, else CPU.
        batch_size:  Eval batch size (default 1024).
        threshold:   Decision threshold for macro-F1 over sigmoid(logits) (default 0.50).
        selector_override: If "random", temporarily replace selection_mod with RandomSelection(temperature).
                           If "shared" or None, evaluate with the model’s current selector.
        temperature: Temperature for RandomSelection (lower -> peakier), if overriding.

    Returns:
        Dict with metrics and basic dims. 'status' is "OK" on success.
    """
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Build CLEAN datamodule & loaders (normalize='none') — MATCH BACKBONE
    dm = _build_dm_clean_for(backbone=backbone, batch_size=batch_size)
    img_dim, txt_dim, lbl_dim = _get_dims_from_dm(dm)
    _, val_loader, test_loader = _build_clean_loaders(dm, dev, batch_size=batch_size)

    # Optional: temporarily override selector
    original_selector = model.selection_mod
    try:
        if selector_override is not None:
            if selector_override not in {"shared", "random"}:
                raise ValueError("selector_override must be one of {'shared','random'} or None.")
            if selector_override == "random":
                model.selection_mod = RandomSelection(temperature=float(temperature))
            # if "shared": keep whatever the model currently has (assumed learned/shared-keys)

        # Inference
        logits_val = _infer_logits(model, val_loader, dev)
        logits_te  = _infer_logits(model, test_loader, dev)
    finally:
        # restore original selector to avoid side effects
        model.selection_mod = original_selector

    y_val = _collect_labels(val_loader).astype(np.float32)
    y_te  = _collect_labels(test_loader).astype(np.float32)

    # Metrics
    bce_val = float(F.binary_cross_entropy_with_logits(
        torch.from_numpy(logits_val), torch.from_numpy(y_val), reduction="mean").item())
    bce_te = float(F.binary_cross_entropy_with_logits(
        torch.from_numpy(logits_te), torch.from_numpy(y_te), reduction="mean").item())

    thr = float(threshold)
    y_pred_val = (_sigmoid_np(logits_val) >= thr).astype(np.int64)
    y_pred_te  = (_sigmoid_np(logits_te)  >= thr).astype(np.int64)

    macro_f1_val = float(f1_score(y_val, y_pred_val, average='macro', zero_division=0))
    macro_f1_te  = float(f1_score(y_te,  y_pred_te,  average='macro', zero_division=0))

    out = dict(
        status="OK",
        img_dim=img_dim, txt_dim=txt_dim, lbl_dim=lbl_dim,
        bce_val=bce_val, bce_test=bce_te,
        macro_f1_val=macro_f1_val, macro_f1_test=macro_f1_te,
        n_val=int(y_val.shape[0]), n_test=int(y_te.shape[0]),
        batch_size=int(batch_size), threshold=float(threshold),
        view=VIEW_MODE_EVAL, normalize=NORMALIZE_FLAG,
        backbone=backbone,
        selector_used=(selector_override or "model"),
        temperature=(float(temperature) if selector_override == "random" else None),
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out


__all__ = ["evaluate_clean", "RandomSelection"]
