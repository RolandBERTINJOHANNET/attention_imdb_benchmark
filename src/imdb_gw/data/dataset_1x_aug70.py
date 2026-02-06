#!/usr/bin/env python3
# dataset_1x_aug70.py — IMDB 1.0 latents datamodule (CLIP/BLIP2 backbones, aug-70 views)
# Adds OPTIONAL labels modality built ON-THE-FLY from base labels:
#  - Loads /home/rbertin/attention/imdb1/labels_23/{labels_all_23.npy, ids_all.txt}
#  - Reorders to extractor's orig_idx (lexicographic dataset stems)
#  - Duplicates each label vector across all 70 views so it index-aligns with image/text.
# Also adds two *mirror* modalities for clean targets:
#  - image_latents_clean, caption_embeddings_clean
#    (replicate view-0 over all 70 views so every augmented row has a matching clean row).

from __future__ import annotations
from typing import Dict, Tuple, Literal, Iterable, Optional, List
from pathlib import Path
import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
OTHER_BACKBONES_DIR = Path(os.getenv(
    "OTHER_BACKBONES_DIR_IMDB1_AUG70",
    "/home/rbertin/attention/imdb1/embeddings_clip_blip2_aug70"
)).resolve()

DEFAULT_CLIP_IMAGE   = OTHER_BACKBONES_DIR / "mm_imdb1_globals_aug70_clip_image.npy"
DEFAULT_CLIP_TEXT    = OTHER_BACKBONES_DIR / "mm_imdb1_globals_aug70_clip_text.npy"
DEFAULT_BLIP2_IMAGE  = OTHER_BACKBONES_DIR / "mm_imdb1_globals_aug70_blip2_image.npy"
DEFAULT_BLIP2_TEXT   = OTHER_BACKBONES_DIR / "mm_imdb1_globals_aug70_blip2_text.npy"

IMAGE_LATENTS_PATH      = os.getenv("MMIMDB_IMAGE_LATENTS", str(DEFAULT_CLIP_IMAGE))
CAPTION_EMBEDDINGS_PATH = os.getenv("MMIMDB_TEXT_EMBEDS",  str(DEFAULT_CLIP_TEXT))

# Base labels directory (23-class, multi-label, one row per ORIGINAL sample)
LABELS_DIR_23 = Path(os.getenv(
    "IMDB1_LABELS_DIR_23",
    "/home/rbertin/attention/imdb1/labels_23"
)).resolve()
LABELS_ALL_23_NPY = LABELS_DIR_23 / "labels_all_23.npy"
IDS_ALL_TXT       = LABELS_DIR_23 / "ids_all.txt"

# IMDB1 raw data root (images/json + split.json)
IMDB1_DIR = Path(os.getenv(
    "MMIMDB1_DIR",
    "/home/rbertin/attention/imdb1/unzipped_imdb/imdb"
)).resolve()
DATASET_DIR = IMDB1_DIR / "dataset"
SPLIT_JSON  = IMDB1_DIR / "split.json"

# ──────────────────────────────────────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────────────────────────────────────
def _as_paths(maybe_paths: Optional[Iterable[str] | str]) -> list[Path]:
    if maybe_paths is None:
        return []
    if isinstance(maybe_paths, (str, Path)):
        return [Path(maybe_paths)]
    return [Path(p) for p in maybe_paths]

def load_npy_as_float32_tensor(file_path: str | Path) -> torch.Tensor:
    """Load .npy → contiguous, writable float32 torch tensor."""
    arr = np.load(str(file_path))
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=True)
    else:
        arr = np.array(arr, dtype=np.float32, copy=True)
    return torch.from_numpy(arr)

def load_and_concat(base_path: str | Path, extra_paths: Optional[Iterable[str] | str] = None) -> torch.Tensor:
    """Load base npy, then (optionally) concatenate extras along dim 0."""
    base = load_npy_as_float32_tensor(base_path)
    base_N, base_D = base.shape
    tensors = [base]
    for p in _as_paths(extra_paths):
        t = load_npy_as_float32_tensor(p)
        if t.ndim != 2 or t.shape[1] != base_D:
            raise ValueError(f"Feature dim mismatch when concatenating '{p}': got {tuple(t.shape)}, expected (_, {base_D})")
        tensors.append(t)
    if len(tensors) == 1:
        return base
    out = torch.cat(tensors, dim=0)
    extra_total = out.shape[0] - base_N
    print(f"Concatenated {len(tensors)-1} extra file(s) to '{base_path}': base N={base_N}, +extra={extra_total} → total N={out.shape[0]}, D={out.shape[1]}")
    return out

def _guess_manifest_path(data_path: Path) -> Optional[Path]:
    name = data_path.name
    parent = data_path.parent
    base = (name
            .replace("_clip_image.npy", "")
            .replace("_clip_text.npy", "")
            .replace("_blip2_image.npy", "")
            .replace("_blip2_text.npy", "")
            .replace("_labels.npy", ""))
    cand = parent / f"{base}_manifest.json"
    if cand.exists():
        return cand
    manifests = sorted(parent.glob("*_manifest.json"))
    if len(manifests) == 1:
        return manifests[0]
    return None

def _load_views_info(any_modality_path: Path, manifest_path: Optional[Path] = None) -> Tuple[int, int]:
    """Return (N_original, VIEWS) by reading manifest; fallback to heuristic."""
    if manifest_path is None:
        manifest_path = _guess_manifest_path(any_modality_path)
    if manifest_path and manifest_path.exists():
        d = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        N_total = int(d["total_rows"])
        N_orig  = int(d["original_rows"])
        V       = int(d["views_per_sample"])
        if N_total != N_orig * V:
            raise ValueError(f"Manifest mismatch: total_rows={N_total} vs original_rows*V={N_orig*V}")
        return N_orig, V

    parent = any_modality_path.parent
    name = any_modality_path.name
    base = (name
            .replace("_clip_image.npy", "")
            .replace("_clip_text.npy", "")
            .replace("_blip2_image.npy", "")
            .replace("_blip2_text.npy", "")
            .replace("_labels.npy", ""))
    index_csv = parent / f"{base}_aug_index.csv"
    data = np.load(str(any_modality_path))
    N_total = int(data.shape[0])
    if index_csv.exists():
        try:
            import pandas as _pd  # optional
            df = _pd.read_csv(index_csv)
            V = int(df["view"].max()) + 1
            max_orig = int(df["orig_idx"].max())
            N_orig = max_orig + 1
            if N_total != N_orig * V:
                raise ValueError("Index CSV implies inconsistent N_total.")
            return N_orig, V
        except Exception:
            pass
    V = 70
    if N_total % V != 0:
        raise ValueError(f"Cannot infer views: N_total={N_total} not divisible by assumed V={V}.")
    N_orig = N_total // V
    return N_orig, V

def _list_by_suffix(root: Path, exts: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    for e in exts:
        out += list(root.glob(f"*{e}"))
        out += list(root.glob(f"*{e.upper()}"))
    out.sort()
    return out

def _sorted_stems_from_dataset(dataset_dir: Path) -> list[str]:
    """Extractor's canonical order: sorted intersection of image/json stems."""
    images = _list_by_suffix(dataset_dir, (".jpg", ".jpeg"))
    jsons  = _list_by_suffix(dataset_dir, (".json",))
    img_map = {p.stem: p for p in images}
    js_map  = {p.stem: p for p in jsons}
    stems = sorted(set(img_map) & set(js_map))
    if not stems:
        raise FileNotFoundError(f"No aligned image/json stems found in {dataset_dir}")
    return stems  # lexicographic

def _normalize_stem_token(x, pad_width: int) -> str:
    if isinstance(x, int):
        return str(x).zfill(pad_width)
    s = str(x)
    return s if len(s) >= pad_width else s.zfill(pad_width)

def _build_indices_from_split_json(
    dataset_dir: Path,
    split_json_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Read split.json {train/dev/test} and map stems → orig indices (lexicographic)."""
    if not split_json_path.exists():
        raise FileNotFoundError(f"split.json not found: {split_json_path}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset dir not found: {dataset_dir}")

    stems_sorted = _sorted_stems_from_dataset(dataset_dir)
    pad_width = max(len(s) for s in stems_sorted)
    stem2idx = {s: i for i, s in enumerate(stems_sorted)}

    sp = json.loads(split_json_path.read_text(encoding="utf-8"))
    train_ids = sp.get("train", [])
    val_ids   = sp.get("dev", sp.get("val", []))
    test_ids  = sp.get("test", [])

    def map_split(ids):
        idxs, missing = [], []
        for tok in ids:
            key = _normalize_stem_token(tok, pad_width)
            i = stem2idx.get(key)
            if i is None:
                missing.append(key)
            else:
                idxs.append(i)
        return np.array(sorted(set(idxs)), dtype=np.int64), missing

    idx_train, miss_tr = map_split(train_ids)
    idx_val,   miss_va = map_split(val_ids)
    idx_test,  miss_te = map_split(test_ids)

    if len(miss_tr) or len(miss_va) or len(miss_te):
        raise ValueError(
            "split.json references stems not found in dataset dir.\n"
            f"  missing train: {len(miss_tr)}  val: {len(miss_va)}  test: {len(miss_te)} "
            f"(up to 5) train: {miss_tr[:5]}  val: {miss_va[:5]}  test: {miss_te[:5]}"
        )

    stats = {
        "n_train": int(idx_train.size),
        "n_val":   int(idx_val.size),
        "n_test":  int(idx_test.size),
        "n_total_stems": len(stems_sorted),
    }
    print(f"[split.json] train={stats['n_train']}  val={stats['n_val']}  test={stats['n_test']}  total={stats['n_total_stems']}")
    return idx_train, idx_val, idx_test, stats

def _map_orig_to_aug_indices(idx_orig: torch.LongTensor, V: int, mode: str, view_k: int = 0) -> torch.LongTensor:
    """Map original-sample indices to augmented-row indices for view selection."""
    if mode == "clean":
        k = 0
        return idx_orig * V + k
    elif mode == "view_k":
        if not (0 <= view_k < V):
            raise ValueError(f"view_k={view_k} out of range [0, {V})")
        return idx_orig * V + int(view_k)
    elif mode == "all":
        arangeV = torch.arange(V, dtype=idx_orig.dtype, device=idx_orig.device)
        return (idx_orig[:, None] * V + arangeV[None, :]).reshape(-1)
    else:
        raise ValueError(f"Unsupported mode for index mapping: {mode}")

# ──────────────────────────────────────────────────────────────────────────────
# Labels: load base, align to extractor order, duplicate across views
# ──────────────────────────────────────────────────────────────────────────────
def _load_labels_base() -> tuple[np.ndarray, List[str]]:
    if not LABELS_ALL_23_NPY.exists() or not IDS_ALL_TXT.exists():
        raise FileNotFoundError(
            f"Labels base files not found:\n  {LABELS_ALL_23_NPY}\n  {IDS_ALL_TXT}\n"
            "Set IMDB1_LABELS_DIR_23 if needed."
        )
    Y_all = np.load(str(LABELS_ALL_23_NPY))  # (N_all, C)
    if Y_all.ndim != 2:
        raise ValueError(f"labels_all_23.npy must be 2D, got {Y_all.shape}")
    with IDS_ALL_TXT.open("r", encoding="utf-8") as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    if len(ids) != Y_all.shape[0]:
        raise RuntimeError(f"ids_all.txt lines ({len(ids)}) != labels rows ({Y_all.shape[0]})")
    return Y_all.astype(np.float32, copy=False), ids

def _build_labels_in_extractor_order(dataset_dir: Path) -> torch.Tensor:
    """Return labels_base as torch.float32 in EXACT orig_idx order used by extractor."""
    Y_all, ids = _load_labels_base()
    stems = _sorted_stems_from_dataset(dataset_dir)  # length N_orig
    idx_map = {sid: i for i, sid in enumerate(ids)}
    # lowercase fallback
    idx_map_lower = {sid.lower(): i for i, sid in enumerate(ids) if sid.lower() not in idx_map}
    idx_map.update(idx_map_lower)

    rows = []
    missing = []
    for s in stems:
        key = s if s in idx_map else s.lower()
        if key not in idx_map:
            missing.append(s)
            if len(missing) <= 5:
                print(f"[labels] missing stem example: {s}")
        else:
            rows.append(idx_map[key])
    if missing:
        raise KeyError(f"{len(missing)} stems from dataset/ not found in ids_all.txt; examples: {missing[:5]}")
    Y_base = torch.from_numpy(Y_all[rows].copy()).to(torch.float32)  # (N_orig, C)
    return Y_base

def _duplicate_labels_across_views(labels_base: torch.Tensor, V: int) -> torch.Tensor:
    """Tile labels across V views so shape becomes (N_orig*V, C)."""
    return labels_base.repeat_interleave(V, dim=0)

# ──────────────────────────────────────────────────────────────────────────────
# Dataset wrappers
# ──────────────────────────────────────────────────────────────────────────────
class DomainDataset(Dataset):
    def __init__(self, domain_data: Dict[str, torch.Tensor]) -> None:
        self.domain_data = domain_data
        lens = [t.shape[0] for t in domain_data.values()]
        if len(set(lens)) != 1:
            raise ValueError(f"Domain tensors have mismatched lengths: {lens}")
        self._len = lens[0]
    def __len__(self) -> int:
        return self._len
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {name: data[index] for name, data in self.domain_data.items()}

class GWDataModule(LightningDataModule):
    def __init__(
        self,
        val_datasets: Dict[frozenset[str], DomainDataset],
        train_datasets: Dict[frozenset[str], DomainDataset],
        test_datasets: Optional[Dict[frozenset[str], DomainDataset]],
        batch_size: int,
        num_workers: int = 9,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.val_datasets = val_datasets
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets or {}
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup_dataloaders(self, datasets: Dict[frozenset[str], DomainDataset], *, shuffle: bool):
        dls: Dict[frozenset[str], DataLoader] = {}
        for domain_set, dataset in datasets.items():
            dls[domain_set] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        return dls

    def train_dataloader(self):
        return CombinedLoader(self.setup_dataloaders(self.train_datasets, shuffle=True), mode="min_size")
    def val_dataloader(self):
        return CombinedLoader(self.setup_dataloaders(self.val_datasets, shuffle=False), mode="min_size")
    def test_dataloader(self):
        return CombinedLoader(self.setup_dataloaders(self.test_datasets, shuffle=False), mode="min_size")

    def get_samples(self, split: Literal["train", "val", "test"], amount: int) -> Dict[frozenset, Dict[str, torch.Tensor]]:
        loader = {"train": self.train_dataloader, "val": self.val_dataloader, "test": self.test_dataloader}[split]()
        batch = next(iter(loader))
        out: Dict[frozenset, Dict[str, torch.Tensor]] = {}
        for dom_set, tensors in batch.items():
            out[dom_set] = {k: v[:amount] for k, v in tensors.items()}
        return out

# ──────────────────────────────────────────────────────────────────────────────
# Builders (aug-70 aware) — labels duplicated across views (no noise)
# Also builds *clean mirrors* for image/text that replicate view-0 across all views.
# ──────────────────────────────────────────────────────────────────────────────
def make_datasets(
    image_latents_path: str | Path = IMAGE_LATENTS_PATH,
    caption_embeddings_path: str | Path = CAPTION_EMBEDDINGS_PATH,
    *,
    extra_image_latents: Optional[Iterable[str] | str] = None,
    extra_caption_embeddings: Optional[Iterable[str] | str] = None,
    include_labels: bool = False,              # ← turn on to add duplicated labels modality
    val_ratio: float = 0.1,                    # API compatibility; ignored for split.json
    seed: int = 42,                            # API compatibility
    normalize: Literal["none", "train_split"] = "none",
    view_mode: Literal["clean", "mean", "all", "view_k"] = "clean",
    view_k: int = 0,
    manifest_path: Optional[str | Path] = None,
) -> Tuple[
    Dict[frozenset[str], DomainDataset],
    Dict[frozenset[str], DomainDataset],
    Dict[frozenset[str], DomainDataset],
]:
    # 0) Sanity
    for p in [image_latents_path, caption_embeddings_path]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Required file not found: {p}")
    if not SPLIT_JSON.exists():
        raise FileNotFoundError(f"Required split.json not found: {SPLIT_JSON}")
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Required dataset folder not found: {DATASET_DIR}")
    if include_labels:
        if not LABELS_ALL_23_NPY.exists() or not IDS_ALL_TXT.exists():
            raise FileNotFoundError(
                "Labels requested but base files are missing:\n"
                f"  {LABELS_ALL_23_NPY}\n  {IDS_ALL_TXT}\n"
                "Set IMDB1_LABELS_DIR_23 if they live elsewhere."
            )

    # 1) Load latents
    image_all = load_and_concat(image_latents_path,      extra_image_latents)      # (N_total, D_img)
    text_all  = load_and_concat(caption_embeddings_path, extra_caption_embeddings) # (N_total, D_txt)

    # 2) Views info
    N_orig, V = _load_views_info(Path(image_latents_path), Path(manifest_path) if manifest_path else None)
    N_total = int(image_all.shape[0])
    if N_total != N_orig * V:
        raise ValueError(f"Counts mismatch: image N_total={N_total} vs N_orig*V={N_orig*V}")
    if text_all.shape[0] != N_total:
        raise ValueError(f"After concat, text N_total={text_all.shape[0]} ≠ image N_total={N_total}")

    # 2a) Build CLEAN mirrors: view-0 replicated over all views → shape (N_total, D)
    def make_clean_full(arr_full: torch.Tensor) -> torch.Tensor:
        D = arr_full.shape[1]
        arr3 = arr_full.view(N_orig, V, D)
        base_clean = arr3[:, 0, :].contiguous()            # (N_orig, D), true clean
        return base_clean.repeat_interleave(V, dim=0)      # (N_total, D), clean per-aug row

    image_all_clean = make_clean_full(image_all)
    text_all_clean  = make_clean_full(text_all)

    # 2b) Labels: base → extractor order → duplicate across views
    if include_labels:
        labels_base = _build_labels_in_extractor_order(DATASET_DIR)            # (N_orig, C)
        labels_all  = _duplicate_labels_across_views(labels_base, V)           # (N_total, C)
        print(f"[labels] base={tuple(labels_base.shape)}  duplicated→all={tuple(labels_all.shape)}")

    print(f"[aug70] Loaded arrays: image={tuple(image_all.shape)}, text={tuple(text_all.shape)}"
          + (f", labels={tuple(labels_all.shape)}" if include_labels else "")
          + f" (N_orig={N_orig}, V={V})")

    # 3) Official split (orig-level)
    idx_tr_np, idx_va_np, idx_te_np, _stats = _build_indices_from_split_json(DATASET_DIR, SPLIT_JSON)
    if max(idx_tr_np.max(initial=0), idx_va_np.max(initial=0), idx_te_np.max(initial=0)) >= N_orig:
        raise ValueError(
            f"Split indices exceed original data size (N_orig={N_orig}). "
            f"max(train)={idx_tr_np.max(initial=-1)}, max(val)={idx_va_np.max(initial=-1)}, max(test)={idx_te_np.max(initial=-1)}."
        )
    idx_tr = torch.from_numpy(idx_tr_np).long()
    idx_va = torch.from_numpy(idx_va_np).long()
    idx_te = torch.from_numpy(idx_te_np).long()

    # 4) View selection / aggregation for BOTH augmented and clean mirrors
    def select_views_aug_and_clean(arr_full: torch.Tensor, arr_clean_full: torch.Tensor):
        D = arr_full.shape[1]
        if view_mode == "mean":
            arr3 = arr_full.view(N_orig, V, D)
            base_clean = arr_clean_full.view(N_orig, V, D)[:, 0, :]  # (N_orig, D)
            a_tr = arr3.index_select(0, idx_tr).mean(dim=1)
            a_va = arr3.index_select(0, idx_va).mean(dim=1)
            a_te = arr3.index_select(0, idx_te).mean(dim=1)
            c_tr = base_clean.index_select(0, idx_tr)
            c_va = base_clean.index_select(0, idx_va)
            c_te = base_clean.index_select(0, idx_te)
            return a_tr, a_va, a_te, c_tr, c_va, c_te

        elif view_mode in ("clean", "view_k", "all"):
            if view_mode == "clean":
                k = 0
                tr_aug = idx_tr * V + k
                va_aug = idx_va * V + k
                te_aug = idx_te * V + k
                tr_clean = tr_aug  # selecting clean or full is identical for view-0
                va_clean = va_aug
                te_clean = te_aug
            elif view_mode == "view_k":
                k = int(view_k)
                tr_aug = idx_tr * V + k
                va_aug = idx_va * V + k
                te_aug = idx_te * V + k
                tr_clean = idx_tr * V + 0   # mirror target is always clean (k=0)
                va_clean = idx_va * V + 0
                te_clean = idx_te * V + 0
            else:  # "all"
                arangeV = torch.arange(V, dtype=idx_tr.dtype, device=idx_tr.device)
                tr_aug = (idx_tr[:, None] * V + arangeV[None, :]).reshape(-1)
                va_aug = (idx_va[:, None] * V + arangeV[None, :]).reshape(-1)
                te_aug = (idx_te[:, None] * V + arangeV[None, :]).reshape(-1)
                # clean indices: view-0 repeated V times for each original
                tr_clean = (idx_tr * V + 0).repeat_interleave(V)
                va_clean = (idx_va * V + 0).repeat_interleave(V)
                te_clean = (idx_te * V + 0).repeat_interleave(V)

            a_tr = arr_full.index_select(0, tr_aug)
            a_va = arr_full.index_select(0, va_aug)
            a_te = arr_full.index_select(0, te_aug)
            c_tr = arr_clean_full.index_select(0, tr_clean)
            c_va = arr_clean_full.index_select(0, va_clean)
            c_te = arr_clean_full.index_select(0, te_clean)
            return a_tr, a_va, a_te, c_tr, c_va, c_te
        else:
            raise ValueError(f"Unsupported view_mode: {view_mode}")

    img_tr, img_val, img_tst, img_tr_clean, img_val_clean, img_tst_clean = \
        select_views_aug_and_clean(image_all, image_all_clean)
    txt_tr, txt_val, txt_tst, txt_tr_clean, txt_val_clean, txt_tst_clean = \
        select_views_aug_and_clean(text_all,  text_all_clean)

    if include_labels:
        # labels mirror == labels themselves (already duplicated across views)
        def select_labels(arr_full: torch.Tensor):
            if view_mode == "mean":
                D = arr_full.shape[1]
                arr3 = arr_full.view(N_orig, V, D)
                return (
                    arr3.index_select(0, idx_tr).mean(dim=1),
                    arr3.index_select(0, idx_va).mean(dim=1),
                    arr3.index_select(0, idx_te).mean(dim=1),
                )
            elif view_mode in ("clean", "view_k", "all"):
                if view_mode == "clean":
                    idx_tr_aug = idx_tr * V + 0
                    idx_va_aug = idx_va * V + 0
                    idx_te_aug = idx_te * V + 0
                elif view_mode == "view_k":
                    idx_tr_aug = idx_tr * V + int(view_k)
                    idx_va_aug = idx_va * V + int(view_k)
                    idx_te_aug = idx_te * V + int(view_k)
                else:
                    arangeV = torch.arange(V, dtype=idx_tr.dtype, device=idx_tr.device)
                    idx_tr_aug = (idx_tr[:, None] * V + arangeV[None, :]).reshape(-1)
                    idx_va_aug = (idx_va[:, None] * V + arangeV[None, :]).reshape(-1)
                    idx_te_aug = (idx_te[:, None] * V + arangeV[None, :]).reshape(-1)
                return (
                    arr_full.index_select(0, idx_tr_aug),
                    arr_full.index_select(0, idx_va_aug),
                    arr_full.index_select(0, idx_te_aug),
                )
            else:
                raise ValueError(f"Unsupported view_mode for labels: {view_mode}")

        lbl_tr, lbl_val, lbl_tst = select_labels(labels_all)

    print(f"[aug70] Split ({view_mode}):")
    print(f"  train image_latents              : {tuple(img_tr.shape)}")
    print(f"  train caption_embeds             : {tuple(txt_tr.shape)}")
    print(f"  train image_latents_clean        : {tuple(img_tr_clean.shape)}")
    print(f"  train caption_embeddings_clean   : {tuple(txt_tr_clean.shape)}")
    print(f"  val   image_latents              : {tuple(img_val.shape)}")
    print(f"  val   caption_embeds             : {tuple(txt_val.shape)}")
    print(f"  val   image_latents_clean        : {tuple(img_val_clean.shape)}")
    print(f"  val   caption_embeddings_clean   : {tuple(txt_val_clean.shape)}")
    print(f"  test  image_latents              : {tuple(img_tst.shape)}")
    print(f"  test  caption_embeds             : {tuple(txt_tst.shape)}")
    print(f"  test  image_latents_clean        : {tuple(img_tst_clean.shape)}")
    print(f"  test  caption_embeddings_clean   : {tuple(txt_tst_clean.shape)}")
    if include_labels:
        print(f"  (labels) train/val/test          : {tuple(lbl_tr.shape)}, {tuple(lbl_val.shape)}, {tuple(lbl_tst.shape)}")

    # 5) Optional re-normalization (per-modality, using train stats)
    if normalize == "train_split":
        def norm_train_split(tr, va, te):
            m, s = tr.mean(dim=0, keepdim=True), tr.std(dim=0, keepdim=True)
            s[s == 0] = 1e-6
            return (tr - m)/s, (va - m)/s, (te - m)/s

        img_tr, img_val, img_tst = norm_train_split(img_tr, img_val, img_tst)
        txt_tr, txt_val, txt_tst = norm_train_split(txt_tr, txt_val, txt_tst)
        img_tr_clean, img_val_clean, img_tst_clean = norm_train_split(img_tr_clean, img_val_clean, img_tst_clean)
        txt_tr_clean, txt_val_clean, txt_tst_clean = norm_train_split(txt_tr_clean, txt_val_clean, txt_tst_clean)
        if include_labels:
            lbl_tr, lbl_val, lbl_tst = norm_train_split(lbl_tr, lbl_val, lbl_tst)
        print("Applied train-split normalization to all included modalities.")
    else:
        print("No extra normalization (using extractor’s dataset-wise z-scoring).")

    # 6) Datasets
    train_datasets = {
        frozenset(["image_latents"]): DomainDataset({"image_latents": img_tr}),
        frozenset(["caption_embeddings"]): DomainDataset({"caption_embeddings": txt_tr}),
        frozenset(["image_latents", "caption_embeddings"]): DomainDataset({
            "image_latents": img_tr,
            "caption_embeddings": txt_tr,
        }),
        # new clean mirrors
        frozenset(["image_latents_clean"]): DomainDataset({"image_latents_clean": img_tr_clean}),
        frozenset(["caption_embeddings_clean"]): DomainDataset({"caption_embeddings_clean": txt_tr_clean}),
        frozenset(["image_latents", "image_latents_clean"]): DomainDataset({
            "image_latents": img_tr,
            "image_latents_clean": img_tr_clean,
        }),
        frozenset(["caption_embeddings", "caption_embeddings_clean"]): DomainDataset({
            "caption_embeddings": txt_tr,
            "caption_embeddings_clean": txt_tr_clean,
        }),
        frozenset(["image_latents", "caption_embeddings", "image_latents_clean", "caption_embeddings_clean"]): DomainDataset({
            "image_latents": img_tr,
            "caption_embeddings": txt_tr,
            "image_latents_clean": img_tr_clean,
            "caption_embeddings_clean": txt_tr_clean,
        }),
    }
    val_datasets = {
        frozenset(["image_latents", "caption_embeddings"]): DomainDataset({
            "image_latents": img_val,
            "caption_embeddings": txt_val,
        }),
        frozenset(["image_latents_clean"]): DomainDataset({"image_latents_clean": img_val_clean}),
        frozenset(["caption_embeddings_clean"]): DomainDataset({"caption_embeddings_clean": txt_val_clean}),
        frozenset(["image_latents", "image_latents_clean"]): DomainDataset({
            "image_latents": img_val,
            "image_latents_clean": img_val_clean,
        }),
        frozenset(["caption_embeddings", "caption_embeddings_clean"]): DomainDataset({
            "caption_embeddings": txt_val,
            "caption_embeddings_clean": txt_val_clean,
        }),
        frozenset(["image_latents", "caption_embeddings", "image_latents_clean", "caption_embeddings_clean"]): DomainDataset({
            "image_latents": img_val,
            "caption_embeddings": txt_val,
            "image_latents_clean": img_val_clean,
            "caption_embeddings_clean": txt_val_clean,
        }),
    }
    test_datasets = {
        frozenset(["image_latents", "caption_embeddings"]): DomainDataset({
            "image_latents": img_tst,
            "caption_embeddings": txt_tst,
        }),
        frozenset(["image_latents_clean"]): DomainDataset({"image_latents_clean": img_tst_clean}),
        frozenset(["caption_embeddings_clean"]): DomainDataset({"caption_embeddings_clean": txt_tst_clean}),
        frozenset(["image_latents", "image_latents_clean"]): DomainDataset({
            "image_latents": img_tst,
            "image_latents_clean": img_tst_clean,
        }),
        frozenset(["caption_embeddings", "caption_embeddings_clean"]): DomainDataset({
            "caption_embeddings": txt_tst,
            "caption_embeddings_clean": txt_tst_clean,
        }),
        frozenset(["image_latents", "caption_embeddings", "image_latents_clean", "caption_embeddings_clean"]): DomainDataset({
            "image_latents": img_tst,
            "caption_embeddings": txt_tst,
            "image_latents_clean": img_tst_clean,
            "caption_embeddings_clean": txt_tst_clean,
        }),
    }

    if include_labels:
        # label-only + useful combos
        train_datasets[frozenset(["labels"])] = DomainDataset({"labels": lbl_tr})
        val_datasets[frozenset(["labels"])]   = DomainDataset({"labels": lbl_val})
        test_datasets[frozenset(["labels"])]  = DomainDataset({"labels": lbl_tst})

        train_datasets[frozenset(["image_latents", "caption_embeddings", "labels"])] = DomainDataset({
            "image_latents": img_tr, "caption_embeddings": txt_tr, "labels": lbl_tr,
        })
        val_datasets[frozenset(["image_latents", "caption_embeddings", "labels"])] = DomainDataset({
            "image_latents": img_val, "caption_embeddings": txt_val, "labels": lbl_val,
        })
        test_datasets[frozenset(["image_latents", "caption_embeddings", "labels"])] = DomainDataset({
            "image_latents": img_tst, "caption_embeddings": txt_tst, "labels": lbl_tst,
        })

        train_datasets[frozenset(["image_latents", "caption_embeddings", "image_latents_clean", "caption_embeddings_clean", "labels"])] = DomainDataset({
            "image_latents": img_tr, "caption_embeddings": txt_tr,
            "image_latents_clean": img_tr_clean, "caption_embeddings_clean": txt_tr_clean,
            "labels": lbl_tr,
        })
        val_datasets[frozenset(["image_latents", "caption_embeddings", "image_latents_clean", "caption_embeddings_clean", "labels"])] = DomainDataset({
            "image_latents": img_val, "caption_embeddings": txt_val,
            "image_latents_clean": img_val_clean, "caption_embeddings_clean": txt_val_clean,
            "labels": lbl_val,
        })
        test_datasets[frozenset(["image_latents", "caption_embeddings", "image_latents_clean", "caption_embeddings_clean", "labels"])] = DomainDataset({
            "image_latents": img_tst, "caption_embeddings": txt_tst,
            "image_latents_clean": img_tst_clean, "caption_embeddings_clean": txt_tst_clean,
            "labels": lbl_tst,
        })

    return train_datasets, val_datasets, test_datasets

def make_datamodule(
    batch_size: int = 2048,
    val_ratio: float = 0.1,   # ignored
    seed: int = 42,           # ignored
    image_latents_path: str | Path = IMAGE_LATENTS_PATH,
    caption_embeddings_path: str | Path = CAPTION_EMBEDDINGS_PATH,
    *,
    extra_image_latents: Optional[Iterable[str] | str] = None,
    extra_caption_embeddings: Optional[Iterable[str] | str] = None,
    include_labels: bool = False,
    num_workers: int = 9,
    pin_memory: bool = True,
    normalize: Literal["none", "train_split"] = "none",
    view_mode: Literal["clean", "mean", "all", "view_k"] = "clean",
    view_k: int = 0,
    manifest_path: Optional[str | Path] = None,
) -> GWDataModule:
    train_datasets, val_datasets, test_datasets = make_datasets(
        image_latents_path=image_latents_path,
        caption_embeddings_path=caption_embeddings_path,
        extra_image_latents=extra_image_latents,
        extra_caption_embeddings=extra_caption_embeddings,
        include_labels=include_labels,
        val_ratio=val_ratio,
        seed=seed,
        normalize=normalize,
        view_mode=view_mode,
        view_k=view_k,
        manifest_path=manifest_path,
    )
    return GWDataModule(
        train_datasets=train_datasets,
        val_datasets=val_datasets,
        test_datasets=test_datasets,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Convenience wrappers
# ──────────────────────────────────────────────────────────────────────────────
def make_datamodule_clip_aug70(**kwargs) -> GWDataModule:
    """CLIP pair on MM-IMDb 1.0 aug-70; optional labels duplicated across views; clean mirrors included."""
    return make_datamodule(
        image_latents_path=str(DEFAULT_CLIP_IMAGE),
        caption_embeddings_path=str(DEFAULT_CLIP_TEXT),
        **kwargs,
    )

def make_datamodule_blip2_aug70(**kwargs) -> GWDataModule:
    """BLIP-2 pair on MM-IMDb 1.0 aug-70; optional labels duplicated across views; clean mirrors included."""
    return make_datamodule(
        image_latents_path=str(DEFAULT_BLIP2_IMAGE),
        caption_embeddings_path=str(DEFAULT_BLIP2_TEXT),
        **kwargs,
    )
