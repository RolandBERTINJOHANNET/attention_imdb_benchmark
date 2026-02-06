#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_mm_imdb1_label_embeddings_aug70.py
---------------------------------------
Creates a third "embedding" file for MM-IMDb containing *augmented labels*,
aligned 1:1 with the CLIP/BLIP2 aug70 embeddings produced by
make_mm_imdb1_clip_blip2_embeddings_aug70_batched.py.

Rules matched exactly:
  • Ordering: row = orig_idx * VIEWS + view, with view=0 clean, 1..69 augmented.
  • orig_idx comes from lexicographic stems of aligned (image,json) pairs.
  • Deterministic augmentation per (orig_idx, view) via SHA-256-derived seed.
  • NO normalization/z-scoring is applied to the labels file.

Augmentation for labels:
  • Base labels are multi-hot vectors (shape = (N, C), C=23).
  • view=0 uses the exact multi-hot vector.
  • view=1..69 add Gaussian noise ε ~ N(0, NOISE_STD^2), then CLIP the noise to ±NOISE_CLIP.
  • Final augmented values are CLAMPED to [0, 1] so they remain valid probabilities.
"""

import os, sys, json, time, hashlib
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm

# ---------------- Config (env-overridable) ----------------
DATA_DIR        = Path(os.getenv("MMIMDB1_DATASET_DIR", "./data/imdb1/dataset"))
OUT_DIR         = Path(os.getenv("EMB_OUT_DIR", "./data/embeddings_clip_blip2_aug70"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PREFIX      = "mm_imdb1_globals_aug70"

LABELS_DIR_23   = Path(os.getenv("IMDB1_LABELS_DIR_23", "./data/labels_23"))
LABELS_ALL_NPY  = LABELS_DIR_23 / "labels_all_23.npy"
IDS_ALL_TXT     = LABELS_DIR_23 / "ids_all.txt"

# Views / batching
VIEWS           = 70      # 0=clean, 1..69=aug
ORIG_BATCH      = 4096    # originals per step (pure CPU work; can be large)

# Label augmentation
BASE_SEED       = 123
NOISE_STD       = 0.10     # std of Gaussian before clipping
NOISE_CLIP      = 0.20     # hard clip of noise to ±0.2
CLAMP_TO_01     = True     # keep augmented labels in [0,1]

# ---------------- Utils ----------------
def create_memmap(path: Path, shape, dtype=np.float32):
    from numpy.lib.format import open_memmap
    return open_memmap(str(path), mode='w+', dtype=dtype, shape=shape)

def out_paths(kind: str):
    base = f"{OUT_PREFIX}_{kind}"
    return OUT_DIR / f"{base}.npy", OUT_DIR / f"{base}.stats.npz"

def list_by_suffix(root: Path, exts: Tuple[str, ...]) -> List[Path]:
    out = []
    for e in exts:
        out += list(root.glob(f"*{e}")); out += list(root.glob(f"*{e.upper()}"))
    out.sort(); return out

def pair_samples_by_stem(root: Path) -> List[Tuple[Path, Path]]:
    images = list_by_suffix(root, (".jpg", ".jpeg"))
    jsons  = list_by_suffix(root, (".json",))
    img_map = {p.stem: p for p in images}
    js_map  = {p.stem: p for p in jsons}
    stems = sorted(set(img_map) & set(js_map))
    if not stems:
        print("❌ No aligned image/json stems."); sys.exit(1)
    if len(stems) < max(len(images), len(jsons)):
        print(f"⚠️ Dropped {max(len(images), len(jsons)) - len(stems)} unmatched files.")
    return [(img_map[s], js_map[s]) for s in stems]

def seed_for(orig_idx: int, view: int, salt: str) -> int:
    h = hashlib.sha256(f"{BASE_SEED}:{orig_idx}:{view}:{salt}".encode()).hexdigest()
    return int(h[:8], 16)

def load_labels_and_ids() -> Tuple[np.ndarray, List[str]]:
    if not LABELS_ALL_NPY.exists() or not IDS_ALL_TXT.exists():
        raise FileNotFoundError("labels_all_23.npy and/or ids_all.txt not found. Build them first.")
    Y_all = np.load(LABELS_ALL_NPY)  # (N_all, C)
    with IDS_ALL_TXT.open("r", encoding="utf-8") as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    if len(ids) != Y_all.shape[0]:
        raise RuntimeError(f"Mismatch: ids_all.txt has {len(ids)} lines, labels has {Y_all.shape[0]} rows.")
    return Y_all.astype(np.float32, copy=False), ids

def build_labels_in_embed_order(pairs: List[Tuple[Path, Path]], Y_all: np.ndarray, ids: List[str]) -> np.ndarray:
    """Reorder labels to the exact orig_idx (=lexicographic stem) used by the embedding script."""
    stem_to_row: Dict[str, int] = {stem: i for i, stem in enumerate(ids)}
    stems_in_pairs = [img.stem for (img, _) in pairs]
    missing = [s for s in stems_in_pairs if s not in stem_to_row]
    if missing:
        raise KeyError(f"{len(missing)} stems from DATA_DIR not found in ids_all.txt, e.g. {missing[:5]}")
    idxs = [stem_to_row[s] for s in stems_in_pairs]
    return Y_all[idxs].copy()  # (N, C) in the SAME order as pairs

# ---------------- Main ----------------
def main():
    t0 = time.time()
    print("=== MM-IMDb 1.0 → LABEL embeddings with 70 aligned views (Gaussian aug, clipped ±0.2, no normalization) ===")
    print(f"📂 DATA_DIR={DATA_DIR}\n🗃️ OUT_DIR={OUT_DIR}\n🧪 NOISE_STD={NOISE_STD}  NOISE_CLIP=±{NOISE_CLIP}  VIEWS={VIEWS}")

    # Orig ordering (must match image/text embedding script)
    pairs = pair_samples_by_stem(DATA_DIR); N = len(pairs)
    print(f"🔗 Aligned pairs: {N}")

    # Load base labels and reorder to the same orig_idx
    Y_all, ids = load_labels_and_ids()
    Y = build_labels_in_embed_order(pairs, Y_all, ids)  # (N, C)
    N, C = Y.shape
    total_rows = N * VIEWS
    print(f"🏷️ Labels: N={N}, C={C}")

    # Output memmap
    p_lbl, p_stats = out_paths("labels")
    mm_lbl = create_memmap(p_lbl, (total_rows, C), np.float32)

    # If an old stats file exists from the previous pipeline, remove it to avoid confusion.
    if p_stats.exists():
        try:
            p_stats.unlink()
            print(f"🧹 Removed legacy stats file: {p_stats}")
        except Exception as e:
            print(f"⚠️ Could not remove legacy stats file ({p_stats}): {e}")

    # We won't overwrite the shared aug_index.csv if it already exists.
    index_csv = OUT_DIR / f"{OUT_PREFIX}_aug_index.csv"
    write_index = not index_csv.exists()
    if write_index:
        with index_csv.open("w", encoding="utf-8") as f:
            f.write("aug_idx,orig_idx,view,stem,image_name,text_len\n")

    prog = tqdm(total=N, desc="Aug70 labels (batched originals)", unit="orig")
    for start in range(0, N, ORIG_BATCH):
        end = min(start + ORIG_BATCH, N); B = end - start
        batch_pairs = pairs[start:end]
        stems = [p[0].stem for p in batch_pairs]
        Ys = Y[start:end]  # (B, C)

        meta_lines: List[str] = []

        # Fill all 70 views for this batch
        for v in range(VIEWS):
            for bi in range(B):
                orig_idx = start + bi
                gidx = orig_idx * VIEWS + v

                if v == 0:
                    yv = Ys[bi]
                else:
                    seed = seed_for(orig_idx, v, "labels")
                    rng = np.random.default_rng(seed)
                    noise = rng.normal(loc=0.0, scale=NOISE_STD, size=(C,)).astype(np.float32)
                    # clip noise to ±NOISE_CLIP, then add
                    if NOISE_CLIP is not None and NOISE_CLIP > 0:
                        noise = np.clip(noise, -NOISE_CLIP, NOISE_CLIP, out=noise)
                    yv = Ys[bi] + noise
                    # clamp final values to [0,1] (keeps probabilities valid; still preserves argmax)
                    if CLAMP_TO_01:
                        np.clip(yv, 0.0, 1.0, out=yv)

                mm_lbl[gidx] = yv

                if write_index:
                    meta_lines.append(f"{gidx},{orig_idx},{v},{stems[bi]},{batch_pairs[bi][0].name},-1\n")

        if write_index and meta_lines:
            with index_csv.open("a", encoding="utf-8") as f:
                f.writelines(meta_lines)

        prog.update(B)
    prog.close()

    # NO normalization / NO z-scoring step.
    print("✅ No normalization applied. Labels written as clean/augmented values (with noise clipped and outputs clamped to [0,1]).")

    # Manifest
    manifest = {
        "total_rows": int(total_rows),
        "original_rows": int(N),
        "views_per_sample": int(VIEWS),
        "dims": {"labels": int(C)},
        "dtype": "float32",
        "labels_noise_std": float(NOISe_STD) if 'NOISe_STD' in globals() else float(NOISE_STD),
        "labels_noise_clip": float(NOISE_CLIP),
        "labels_clamped_to_01": bool(CLAMP_TO_01),
        "index_csv": str(index_csv),
        "source_labels_npy": str(LABELS_ALL_NPY),
        "source_ids_txt": str(IDS_ALL_TXT),
        "normalized": False
    }
    (OUT_DIR / f"{OUT_PREFIX}_labels_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("✅ Done.")
    print(f"• {p_lbl.name} shape={mm_lbl.shape}\n• Index CSV: {index_csv}")
    print(f"⏱️  Elapsed: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    import time
    main()
