"""
Local monkeypatches for shimmer to match private diffs.
Apply by importing this module before constructing models.
"""
from __future__ import annotations

from collections.abc import Mapping, Iterable, Callable
from pathlib import Path
from typing import Any

import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR
from torch.nn import Module

from shimmer import modules as _m
from shimmer.modules.global_workspace import (
    GlobalWorkspaceBase,
    SchedulerArgs,
    freeze_domain_modules,
)
from shimmer.modules.gw_module import GWModule
from shimmer.modules.contrastive_loss import ContrastiveLoss, ContrastiveLossType
from shimmer.modules.domain import DomainModule
from shimmer.modules.selection import SelectionBase
from shimmer.modules.losses import BroadcastLossCoefs, GWLosses

# ---- Selection patches ----

class DynamicQueryAttentionOnGWLatents(SelectionBase):
    """Minimal K–Q attention over GW latents with optional time-pos."""

    def __init__(self, head_size: int, gw_dim: int, domain_names: Iterable[str], n_steps: int = 1, use_time_pos: bool = True):
        super().__init__()
        self.head_size = int(head_size)
        self.gw_dim = int(gw_dim)
        self.domain_names = list(domain_names)
        self.n_steps = int(n_steps)
        self.use_time_pos = bool(use_time_pos)

        self.query_layer = torch.nn.Linear(self.gw_dim, self.head_size)
        self.shared_key_layer = torch.nn.Linear(self.gw_dim, self.head_size)
        self.step_pos = torch.nn.Embedding(self.n_steps + 1, self.head_size) if self.use_time_pos else None

    @staticmethod
    def _weighted_sum(gw_latents: dict[str, torch.Tensor], attn: dict[str, torch.Tensor]) -> torch.Tensor:
        out = None
        for d, w in attn.items():
            if d not in gw_latents:
                continue
            term = gw_latents[d] * w.unsqueeze(1)
            out = term if out is None else out + term
        return out

    def _calc_attention(self, keys: dict[str, torch.Tensor], query: torch.Tensor) -> dict[str, torch.Tensor]:
        names = [d for d in self.domain_names if d in keys]
        dots = [(keys[d] * query).sum(dim=1) for d in names]
        logits = torch.stack(dots, dim=1)
        probs = torch.softmax(logits, dim=1)
        return {d: probs[:, i] for i, d in enumerate(names)}

    def forward(self, gw_latents: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        first = next(iter(gw_latents.values()))
        B, device = first.shape[0], first.device

        base_keys = {d: self.shared_key_layer(gw_latents[d]) for d in self.domain_names if d in gw_latents}
        q_in = torch.stack([gw_latents[d] for d in self.domain_names if d in gw_latents], dim=0).mean(0)
        query = self.query_layer(q_in)

        if self.step_pos is not None:
            pe0 = self.step_pos(torch.zeros(B, dtype=torch.long, device=device))
            attn = self._calc_attention({d: base_keys[d] + pe0 for d in base_keys}, query + pe0)
        else:
            attn = self._calc_attention(base_keys, query)

        for s in range(1, self.n_steps + 1):
            fused = self._weighted_sum(gw_latents, attn)
            query = self.query_layer(fused)
            if self.step_pos is not None:
                pes = self.step_pos(torch.full((B,), s, dtype=torch.long, device=device))
                attn = self._calc_attention({d: base_keys[d] + pes for d in base_keys}, query + pes)
            else:
                attn = self._calc_attention(base_keys, query)
        return attn

    def __call__(self, encodings, gw_latents: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.forward(gw_latents)


class ContentQ0SharedKeysSingleStep(SelectionBase):
    """Single-step dot-product softmax selector with shared or per-domain keys."""

    def __init__(self, gw_dim: int, domain_names: Iterable[str], head_size: int = 64, per_domain_keys: bool = False):
        super().__init__()
        self.gw_dim = int(gw_dim)
        self.head_size = int(head_size)
        self.domain_names = list(domain_names)
        self.per_domain_keys = bool(per_domain_keys)

        self.query_layer = torch.nn.Linear(self.gw_dim, self.head_size)
        self.shared_key_layer = torch.nn.Linear(self.gw_dim, self.head_size)
        self.per_key_layers = torch.nn.ModuleDict({d: torch.nn.Linear(self.gw_dim, self.head_size) for d in self.domain_names})

    @staticmethod
    def _calc_attention(keys: dict[str, torch.Tensor], query: torch.Tensor, order: Iterable[str]) -> dict[str, torch.Tensor]:
        names = [d for d in order if d in keys]
        logits = torch.stack([(keys[d] * query).sum(dim=1) for d in names], dim=1)
        probs = torch.softmax(logits, dim=1)
        return {d: probs[:, i] for i, d in enumerate(names)}

    def forward(self, gw_latents: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        present = [d for d in self.domain_names if d in gw_latents]
        if not present:
            raise ValueError("ContentQ0SharedKeysSingleStep: no known domains present in gw_latents.")
        if self.per_domain_keys:
            base_keys = {d: self.per_key_layers[d](gw_latents[d]) for d in present}
        else:
            proj = self.shared_key_layer
            base_keys = {d: proj(gw_latents[d]) for d in present}
        q_in = torch.stack([gw_latents[d] for d in present], dim=0).mean(0)
        query = self.query_layer(q_in)
        return self._calc_attention(base_keys, query, self.domain_names)

    def __call__(self, encodings, gw_latents: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.forward(gw_latents)


# ---- Patched GlobalWorkspaceFusion ----

class GlobalWorkspaceFusion(
    GlobalWorkspaceBase[GWModule, DynamicQueryAttentionOnGWLatents, GWLosses]
):
    """Fusion GW using trainable attention selector (replaces RandomSelection)."""

    def __init__(
        self,
        domain_mods: Mapping[str, DomainModule],
        gw_encoders: Mapping[str, Module],
        gw_decoders: Mapping[str, Module],
        workspace_dim: int,
        loss_coefs: BroadcastLossCoefs | Mapping[str, float],
        # Trainable selector hyperparams
        attn_head_size: int = 256,
        attn_n_steps: int = 1,
        attn_use_time_pos: bool = True,
        # Legacy / compat
        selection_temperature: float = 0.1,
        # Optim/scheduler/contrastive
        optim_lr: float = 1e-3,
        optim_weight_decay: float = 0.0,
        scheduler_args: SchedulerArgs | None = None,
        learn_logit_scale: bool = False,
        contrastive_loss: ContrastiveLossType | None = None,
        scheduler: Callable[[Optimizer], LRScheduler] | None | _m.global_workspace.OneCycleSchedulerSentinel = _m.global_workspace.OneCycleSchedulerSentinel.DEFAULT,
        fusion_activation_fn: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
        per_domain_keys: bool = False,
    ) -> None:
        self.save_hyperparameters(
            {
                "attn_head_size": attn_head_size,
                "attn_n_steps": attn_n_steps,
                "attn_use_time_pos": attn_use_time_pos,
                "selection_temperature": selection_temperature,
                "workspace_dim": workspace_dim,
                "optim_lr": optim_lr,
                "optim_weight_decay": optim_weight_decay,
                "learn_logit_scale": learn_logit_scale,
            }
        )
        self.selection_temperature_deprecated = selection_temperature

        domain_mods = freeze_domain_modules(domain_mods)
        gw_mod = GWModule(domain_mods, workspace_dim, gw_encoders, gw_decoders, fusion_activation_fn)
        if contrastive_loss is None:
            contrastive_loss = ContrastiveLoss(torch.tensor([1 / 0.07]).log(), "mean", learn_logit_scale)
        selection_mod = ContentQ0SharedKeysSingleStep(
            head_size=attn_head_size,
            gw_dim=workspace_dim,
            domain_names=tuple(domain_mods.keys()),
            per_domain_keys=per_domain_keys,
        )
        loss_mod = GWLosses(gw_mod, selection_mod, domain_mods, loss_coefs, contrastive_loss)

        super().__init__(gw_mod, selection_mod, loss_mod, optim_lr, optim_weight_decay, scheduler_args, scheduler)


# ---- Loss patches ----

def contrastive_loss_patched(
    gw_mod: _m.gw_module.GWModuleBase,
    latent_domains: _m.losses.LatentsDomainGroupsT,
    contrastive_fn: ContrastiveLossType,
) -> dict[str, torch.Tensor]:
    """Contrastive loss that ignores *_clean modalities and is zero-safe."""
    losses: dict[str, torch.Tensor] = {}
    metrics: dict[str, torch.Tensor] = {}
    for latents in latent_domains.values():
        latents_cont = {k: v for k, v in latents.items() if not k.endswith("_clean")}
        if len(latents_cont) < 2:
            continue
        cont_latents = gw_mod.encode(latents_cont)
        seen_pairs: set[frozenset[str]] = set()
        key_suffix = "_".join(sorted(latents_cont.keys()))
        items = list(cont_latents.items())
        for i, (d1, z1) in enumerate(items):
            for j in range(i + 1, len(items)):
                d2, z2 = items[j]
                pair = frozenset((d1, d2))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                loss_name = f"contrastive_{d1}_and_{d2}_keys_{key_suffix}"
                out = contrastive_fn(z1, z2)
                losses[loss_name] = out.loss
                metrics.update({f"{loss_name}_{k}": v for k, v in out.metrics.items()})
    if losses:
        losses["contrastives"] = torch.stack(list(losses.values()), dim=0).mean()
    else:
        dev = None
        for group in latent_domains.values():
            if group:
                dev = next(iter(group.values())).device
                break
        losses["contrastives"] = torch.tensor(0.0, device=dev)
    losses.update(metrics)
    return losses


from shimmer.modules.losses import demi_cycle_loss, cycle_loss, translation_loss, generate_partitions
from shimmer.modules.selection import RandomSelection

def broadcast_loss_patched(
    gw_mod: _m.gw_module.GWModuleBase,
    selection_mod: SelectionBase,
    domain_mods: Mapping[str, DomainModule],
    latent_domains: _m.losses.LatentsDomainGroupsT,
    raw_data: _m.losses.RawDomainGroupsT,
) -> dict[str, torch.Tensor]:
    """Broadcast loss with clean-target handling and required modality set."""
    losses: dict[str, torch.Tensor] = {}
    metrics: dict[str, torch.Tensor] = {}
    demi_cycle_losses: list[str] = []
    cycle_losses: list[str] = []
    translation_losses: list[str] = []
    fused_losses: list[str] = []
    rand_sel = RandomSelection(temperature=1.0)
    required = frozenset({"image_latents", "caption_embeddings", "labels", "image_latents_clean", "caption_embeddings_clean"})
    valid_groups = [(gd, lt) for gd, lt in latent_domains.items() if frozenset(gd) == required]
    if not valid_groups:
        raise ValueError("broadcast_loss: no group matches required modalities")
    for group_domains, latents in valid_groups:
        active = ["image_latents", "caption_embeddings", "labels"]
        latents_active = {k: latents[k] for k in active}

        def _gt_keys(d: str) -> tuple[str, str]:
            if d in ("image_latents", "caption_embeddings"):
                return f"{d}_clean", f"{d}_clean"
            return d, d

        encoded_active = gw_mod.encode(latents_active)
        partitions = generate_partitions(len(active))
        group_name = "{" + ",".join(sorted(active)) + "}"
        for partition in partitions:
            selected_latents = {domain: latents_active[domain] for domain, present in zip(active, partition, strict=True) if present}
            selected_encoded_latents = {domain: encoded_active[domain] for domain in selected_latents}
            selected_group_label = "{" + ",".join(sorted(selected_latents)) + "}"
            selected_keys = set(selected_latents.keys())
            if selected_keys == {"image_latents", "caption_embeddings"}:
                selection_scores = selection_mod(selected_latents, selected_encoded_latents)
            else:
                selection_scores = rand_sel(selected_latents, selected_encoded_latents)
            fused_latents = gw_mod.fuse(selected_encoded_latents, selection_scores)
            decoded_latents = gw_mod.decode(fused_latents, domains=active)
            num_active_domains = sum(partition)
            num_total_domains = len(decoded_latents)
            for domain, pred in decoded_latents.items():
                if domain not in active:
                    continue
                latent_key, raw_key = _gt_keys(domain)
                ground_truth = latents[latent_key]
                raw_gt = raw_data[group_domains][raw_key]
                if num_active_domains == 1 and domain in selected_latents:
                    loss_fn = domain_mods[domain].compute_dcy_loss
                elif domain not in selected_latents:
                    loss_fn = domain_mods[domain].compute_tr_loss
                else:
                    loss_fn = domain_mods[domain].compute_fused_loss
                loss_output = loss_fn(pred, ground_truth, raw_gt)
                if loss_output is None:
                    continue
                loss_label = f"from_{selected_group_label}_to_{domain}"
                losses[loss_label + "_loss"] = loss_output.loss
                metrics.update({f"{loss_label}_{k}": v for k, v in loss_output.metrics.items()})
                if num_active_domains == 1 and domain in selected_latents:
                    demi_cycle_losses.append(loss_label + "_loss")
                elif domain not in selected_latents:
                    translation_losses.append(loss_label + "_loss")
                else:
                    fused_losses.append(loss_label + "_loss")
            if num_active_domains < num_total_domains:
                inverse_selected_latents = {d: decoded_latents[d] for d in decoded_latents if d not in selected_latents}
                inverse_selected_group_label = "{" + ",".join(sorted(inverse_selected_latents)) + "}"
                re_encoded_latents = gw_mod.encode(inverse_selected_latents)
                inverse_keys = set(inverse_selected_latents.keys())
                if inverse_keys == {"image_latents", "caption_embeddings"}:
                    re_selection_scores = selection_mod(inverse_selected_latents, re_encoded_latents)
                else:
                    re_selection_scores = rand_sel(inverse_selected_latents, re_encoded_latents)
                re_fused_latents = gw_mod.fuse(re_encoded_latents, re_selection_scores)
                re_decoded_latents = gw_mod.decode(re_fused_latents, domains=selected_latents.keys())
                for domain in selected_latents:
                    latent_key, raw_key = _gt_keys(domain)
                    re_ground_truth = latents[latent_key]
                    re_raw_gt = raw_data[group_domains][raw_key]
                    re_loss_output = domain_mods[domain].compute_cy_loss(re_decoded_latents[domain], re_ground_truth, re_raw_gt)
                    if re_loss_output is None:
                        continue
                    loss_label = f"from_{selected_group_label}_through_{inverse_selected_group_label}_to_{domain}_case_{group_name}"
                    losses[loss_label + "_loss"] = re_loss_output.loss
                    metrics.update({f"{loss_label}_{k}": v for k, v in re_loss_output.metrics.items()})
                    cycle_losses.append(loss_label + "_loss")
    if demi_cycle_losses:
        metrics["demi_cycles"] = torch.mean(torch.stack([losses[x] for x in demi_cycle_losses]))
    if cycle_losses:
        metrics["cycles"] = torch.mean(torch.stack([losses[x] for x in cycle_losses]))
    if translation_losses:
        metrics["translations"] = torch.mean(torch.stack([losses[x] for x in translation_losses]))
    if fused_losses:
        metrics["fused"] = torch.mean(torch.stack([losses[x] for x in fused_losses]))
    metrics.update(losses)
    return metrics


# Patch GWLosses to use the patched contrastive/broadcast
from shimmer.modules.losses import GWLosses as _GWLosses

class GWLossesPatched(_GWLosses):
    def contrastive_loss(self, latent_domains):
        return contrastive_loss_patched(self.gw_mod, latent_domains, self.contrastive_fn)

    def broadcast_loss(self, latent_domains, raw_data):
        return broadcast_loss_patched(self.gw_mod, self.selection_mod, self.domain_mods, latent_domains, raw_data)


# Apply monkeypatches
_m.global_workspace.GlobalWorkspaceFusion = GlobalWorkspaceFusion
_m.selection.DynamicQueryAttentionOnGWLatents = DynamicQueryAttentionOnGWLatents
_m.selection.ContentQ0SharedKeysSingleStep = ContentQ0SharedKeysSingleStep
_m.losses.GWLosses = GWLossesPatched
_m.losses.contrastive_loss = contrastive_loss_patched
_m.losses.broadcast_loss = broadcast_loss_patched

__all__ = [
    "GlobalWorkspaceFusion",
    "DynamicQueryAttentionOnGWLatents",
    "ContentQ0SharedKeysSingleStep",
    "GWLossesPatched",
    "contrastive_loss_patched",
    "broadcast_loss_patched",
]

# Patch DynamicQueryAttention to accept gw_dim for query projection
from shimmer.modules.selection import DynamicQueryAttention as _DynQA

class DynamicQueryAttentionPatched(_DynQA):
    def __init__(self, head_size: int, gw_dim: int, domain_dim: int, domain_names: Iterable[str], n_steps: int = 1):
        # Note: base class signature lacked gw_dim; we repurpose it for query_layer/init state
        super(_DynQA, self).__init__()
        self.gw_dim = gw_dim
        self.head_size = head_size
        self.query_layer = torch.nn.Linear(self.gw_dim, head_size)
        self.key_layers = torch.nn.ModuleDict({domain: torch.nn.Linear(domain_dim, head_size) for domain in domain_names})
        self.n_steps = n_steps
        self.step_limit = n_steps
        self.register_buffer("initial_gw_state", torch.rand(self.gw_dim))

    def forward(self, domains, encodings_pre_fusion):
        # Reuse the helper defined above
        keys = {domain: self.key_layers[domain](encoding) for domain, encoding in domains.items()}
        batch_size = next(iter(domains.values())).shape[0]
        device = next(iter(domains.values())).device
        query = self.query_layer(self.initial_gw_state.expand(batch_size, -1).to(device))
        attention_dict = ContentQ0SharedKeysSingleStep._calc_attention(keys, query, domains.keys())
        if self.n_steps > 0:
            for _ in range(min(self.step_limit, self.n_steps)):
                fused = DynamicQueryAttentionOnGWLatents._weighted_sum(encodings_pre_fusion, attention_dict)
                query = self.query_layer(fused)
                attention_dict = ContentQ0SharedKeysSingleStep._calc_attention(keys, query, domains.keys())
        return attention_dict

_m.selection.DynamicQueryAttention = DynamicQueryAttentionPatched

__all__.append("DynamicQueryAttentionPatched")
