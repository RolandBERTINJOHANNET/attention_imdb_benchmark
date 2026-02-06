#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
function_train_gw_1x.py
----------------------
Importable utility to instantiate & train ONE GlobalWorkspaceFusion model
on MM-IMDb 1.0 (aug-70), view_mode="all", using the exact configuration from
the 1× runs — H=384, easy OneCycle.

Selector modes:
  • "shared"      -> default attentional selector (per_domain_keys=False)
  • "fixedshared" -> equal-share selector (1/N per present domain; no temperature)

Backward-compat:
  • Passing "random" aliases to "fixedshared" (equal-share), with a console notice.

Usage:
    from function_train_gw_1x import instantiate_and_train
    model = instantiate_and_train(seed=42, backbone="blip2", selector="shared")
    model = instantiate_and_train(seed=42, backbone="clip",  selector="fixedshared")
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

# ==== Your codebase bits ======================================================
from imdb_gw.overrides import shimmer_patches as _sp  # apply shimmer monkeypatches
from shimmer import BroadcastLossCoefs
from shimmer.modules.global_workspace import GlobalWorkspaceFusion
from shimmer.modules.selection import RandomSelection
from imdb_gw.domains.domains_obb import (
    BasicImageDomain,
    TextDomain,
    LabelDomain,
    CleanTargetDomain,
)

# IMPORTANT: import BOTH builders and select by backbone
from imdb_gw.data.dataset_1x_aug70 import (
    make_datamodule_blip2_aug70,
    make_datamodule_clip_aug70,
)

# ============================================================
# Fixed defaults (faithful to the previous 1× scripts)
# ============================================================
VIEW_MODE       = "all"
INCLUDE_LABELS  = True

BATCH_SIZE      = 512
MAX_STEPS       = 500            # (kept as in your snippet)
VAL_CHECK       = 0.99
PRECISION       = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"

# Optim
LR              = 5.38e-4
WEIGHT_DECAY    = 1.7e-5
GRAD_CLIP       = 0.0

# Workspace / stacks
WORKSPACE_DIM   = 512
N_LAYERS        = 2
DROPOUT         = 0.1
HIDDEN_DIM      = 384  # H=384

# Logging / checkpoints (local only; W&B removed)
BASE_LOG_DIR    = Path("./logs_final_1x_gw_models_atttest")
BASE_SAVE_DIR   = Path("./final_1x_gw_models_atttest")

SCHED_TAG       = "sched=onecycle_easy_dv10_fdv10_ps0p2"

# ============================================================
# Scheduler (easy OneCycle, winning recipe for H=384)
# ============================================================
def _sched_onecycle(total_steps: int, base_lr: float, div_factor: float, final_div_factor: float, pct_start: float):
    def factory(optim):
        return torch.optim.lr_scheduler.OneCycleLR(
            optim, max_lr=base_lr, total_steps=total_steps,
            pct_start=pct_start, div_factor=div_factor,
            final_div_factor=final_div_factor, anneal_strategy='cos'
        )
    return factory

SCHED_FACTORY = lambda steps: _sched_onecycle(steps, LR, div_factor=10.0, final_div_factor=10.0, pct_start=0.2)

# ============================================================
# Tiny MLP stacks (as in previous scripts)
# ============================================================
class ResMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1  = nn.Linear(dim, hidden)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2  = nn.Linear(hidden, dim)
    def forward(self, x):
        h = self.norm(x)
        h = self.fc1(h); h = self.act(h); h = self.drop(h); h = self.fc2(h)
        return x + h

def make_encoder(in_dim: int, workspace_dim: int, n_layers: int, hidden: int, dropout: float) -> nn.Sequential:
    layers = [nn.Linear(in_dim, workspace_dim), nn.GELU()]
    for _ in range(n_layers):
        layers.append(ResMLPBlock(workspace_dim, hidden, dropout))
    return nn.Sequential(*layers)

def make_decoder(out_dim: int, workspace_dim: int, n_layers: int, hidden: int, dropout: float) -> nn.Sequential:
    layers: List[nn.Module] = []
    for _ in range(n_layers):
        layers.append(ResMLPBlock(workspace_dim, hidden, dropout))
    layers.append(nn.Linear(workspace_dim, out_dim))  # labels decoder outputs logits
    return nn.Sequential(*layers)


# ============================================================
# Datamodule factory (CHOOSES FEATURES BY BACKBONE)
# ============================================================
def _make_dm_for(backbone: str, **dm_kwargs):
    b = backbone.lower()
    if b == "blip2":
        return make_datamodule_blip2_aug70(**dm_kwargs)
    elif b == "clip":
        return make_datamodule_clip_aug70(**dm_kwargs)
    else:
        raise ValueError(f"Unknown backbone '{backbone}' (expected 'clip' or 'blip2').")

# ============================================================
# Public: instantiate & train (selector ∈ {"shared","fixedshared"})
# ============================================================
def instantiate_and_train(
    seed: int,
    backbone: str,
    selector: str = "shared",
    temperature: float = .1,   # kept for backward-compat; unused for fixedshared
) -> GlobalWorkspaceFusion:
    """
    Instantiate and train a single GW model for the given seed and backbone tag.

    Selector modes:
        - "shared": default attentional selector with shared keys (per_domain_keys=False)
        - "fixedshared": equal-share selector (1/N per present domain; no temperature)

    Backward-compat:
        - "random" is accepted and aliased to "fixedshared" (prints a notice).
    """

    # Repro
    seed_everything(seed, workers=True)

    # Datamodule (features match the backbone; normalize='none' per extractor’s z-scoring)
    dm_kwargs = dict(batch_size=BATCH_SIZE, num_workers=9, pin_memory=True,
                     normalize="none", view_mode=VIEW_MODE, include_labels=INCLUDE_LABELS)
    dm = _make_dm_for(backbone, **dm_kwargs)

    # Shapes
    pair_key = frozenset(["image_latents", "caption_embeddings"])
    lab_key  = frozenset(["labels"])
    train_pair = dm.train_datasets[pair_key]
    img_dim = int(train_pair.domain_data["image_latents"].shape[1])
    txt_dim = int(train_pair.domain_data["caption_embeddings"].shape[1])
    lbl_dim = int(dm.train_datasets[lab_key].domain_data["labels"].shape[1])

    # Domains (clean targets supervision-only)
    domain_mods: Dict[str, nn.Module] = {
        "image_latents":            BasicImageDomain(latent_dim=img_dim),
        "caption_embeddings":       TextDomain(latent_dim=txt_dim),
        "image_latents_clean":      CleanTargetDomain(latent_dim=img_dim, name="image_latents_clean"),
        "caption_embeddings_clean": CleanTargetDomain(latent_dim=txt_dim, name="caption_embeddings_clean"),
        "labels":                   LabelDomain(num_classes=lbl_dim),
    }

    # Encoders / Decoders
    gw_encoders = {
        "image_latents":      make_encoder(img_dim, WORKSPACE_DIM, N_LAYERS, HIDDEN_DIM, DROPOUT),
        "caption_embeddings": make_encoder(txt_dim, WORKSPACE_DIM, N_LAYERS, HIDDEN_DIM, DROPOUT),
        "labels":             make_encoder(lbl_dim, WORKSPACE_DIM, N_LAYERS, HIDDEN_DIM, DROPOUT),
    }
    gw_decoders = {
        "image_latents":      make_decoder(img_dim, WORKSPACE_DIM, N_LAYERS, HIDDEN_DIM, DROPOUT),
        "caption_embeddings": make_decoder(txt_dim, WORKSPACE_DIM, N_LAYERS, HIDDEN_DIM, DROPOUT),
        "labels":             make_decoder(lbl_dim, WORKSPACE_DIM, N_LAYERS, HIDDEN_DIM, DROPOUT),
    }

    # Loss mix
    loss_coefs = BroadcastLossCoefs(
        translations=1.0, demi_cycles=0.5, cycles=0.5, contrastives=0.05, fused=1.0
    )

    # Run name + dirs (use selector tag directly)
    keys_tag = selector  # "shared" or "fixedshared"
    run_name = f"final1x__{backbone}__{VIEW_MODE}__gwH{HIDDEN_DIM}__s{seed}__{SCHED_TAG}__keys={keys_tag}"
    LOG_DIR  = BASE_LOG_DIR  / backbone / run_name
    CKPT_DIR = BASE_SAVE_DIR / backbone / run_name
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # Model: attentional selector with shared keys by default (per_domain_keys=False)
    model = GlobalWorkspaceFusion(
        domain_mods=domain_mods,
        gw_encoders=gw_encoders,
        gw_decoders=gw_decoders,
        workspace_dim=WORKSPACE_DIM,
        loss_coefs=loss_coefs,
        optim_lr=LR,
        optim_weight_decay=WEIGHT_DECAY,
        scheduler=SCHED_FACTORY(MAX_STEPS),  # callable: (optimizer) -> Scheduler
        scheduler_args=None,
        per_domain_keys=False,
    )


    # normalize selector (alias "random" -> "fixedshared")
    if selector == "random":
        model.selection_mod = RandomSelection(temperature=float(temperature))


    if selector not in {"shared", "random"}:
        raise ValueError("selector must be one of {'shared', 'fixedshared'}")

    # Loggers (CSV only)
    csv_logger = CSVLogger(save_dir=str(LOG_DIR), name=run_name, flush_logs_every_n_steps=50)
    loggers = [csv_logger]

    # Callbacks — save ONLY 'last.ckpt' to final folder
    lr_cb  = LearningRateMonitor(logging_interval="step")
    ckpt_cb = ModelCheckpoint(
        dirpath=str(CKPT_DIR),
        filename="last",         # writes last.ckpt
        save_last=True,
        save_top_k=0,            # disable top-k; only last
        monitor=None,
    )

    # Trainer
    trainer = Trainer(
        logger=loggers,
        callbacks=[lr_cb, ckpt_cb],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_steps=MAX_STEPS,
        log_every_n_steps=50,
        #val_check_interval=VAL_CHECK,
        num_sanity_val_steps=0,
        detect_anomaly=False,
        gradient_clip_val=GRAD_CLIP,
        precision=PRECISION,
        enable_checkpointing=True,
    )

    # Sanity print (helps catch wrong feature family)
    print(f"\n[FINAL IMDB1 aug-70] RUN={run_name}")
    print(f"Backbone={backbone} | H={HIDDEN_DIM} | {SCHED_TAG} | keys={keys_tag} | per_domain_keys=False")
    print(f"Feature dims: image_latents={img_dim}, caption_embeddings={txt_dim}, labels={lbl_dim}")

    trainer.fit(model=model, datamodule=dm)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model


__all__ = ["instantiate_and_train", "FixedSharedSelection"]
