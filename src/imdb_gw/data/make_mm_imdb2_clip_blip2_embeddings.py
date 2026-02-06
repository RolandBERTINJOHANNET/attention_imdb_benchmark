#!/usr/bin/env python3
# make_mm_imdb2_clip_blip2_embeddings.py
# --------------------------------------
# Single-vector (pooled) backbone representations for MM-IMDb 2.0, matching CoMM backbones:
#   • CLIP (image):  "openai/clip-vit-base-patch32" → get_image_features()  -> [B, 512]
#   • CLIP (text):   "sentence-transformers/clip-ViT-B-32-multilingual-v1"  -> pooled sentence [B, 512]
#   • BLIP-2 (image): Blip2 Q-Former last_hidden_state mean over Q         -> [B, 768]
#   • BLIP-2 (text):  Blip2 LM encoder last_hidden_state mean over tokens  -> [B, H] (e.g., 2048 for flan-t5-xl)
#
# IMPORTANT: No augmentations (no token masking, no prompts), no L2. After extraction, perform
# dataset-wise Z-SCORE per dimension: Z = (X - mu) / (sigma + 1e-6). Save both the z-scored .npy and the stats.
#
# Outputs (z-scored):
#   • <prefix>_clip_image.npy   → (N, 512)    + <prefix>_clip_image.stats.npz   (mu, sigma)
#   • <prefix>_clip_text.npy    → (N, 512)    + <prefix>_clip_text.stats.npz    (mu, sigma)
#   • <prefix>_blip2_image.npy  → (N, 768)    + <prefix>_blip2_image.stats.npz  (mu, sigma)
#   • <prefix>_blip2_text.npy   → (N, H)      + <prefix>_blip2_text.stats.npz   (mu, sigma)
#
# Run:
#   python make_mm_imdb2_clip_blip2_embeddings.py            # runs ALL four dumps (z-scored)
#   python make_mm_imdb2_clip_blip2_embeddings.py --mode clip_image
#   python make_mm_imdb2_clip_blip2_embeddings.py --mode clip_text
#   python make_mm_imdb2_clip_blip2_embeddings.py --mode blip2_image
#   python make_mm_imdb2_clip_blip2_embeddings.py --mode blip2_text
#
# Env:
#   MMIMDB_DIR=/path/to/imdb/dataset
#   BLIP2_MODEL_ID="Salesforce/blip2-flan-t5-xl"  # or "Salesforce/blip2-opt-1.3b"
#   BATCH_SIZE=32        # start small; auto back-off on OOM
#   TXT_BATCH=32
#   MAX_ITEMS=100        # optional smoke test
#
# Deps:
#   pip install -U torch torchvision pillow numpy tqdm timm sentence-transformers "transformers>=4.53"

import os, sys, gc, json, argparse, traceback
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import torch

# ---------------- Config ----------------
DATA_DIR = Path(os.environ.get("MMIMDB_DIR", "./data/imdb2/unzipped_imdb/imdb/dataset"))

CLIP_MODEL_ID_IMG = "openai/clip-vit-base-patch32"  # pooled image features
CLIP_TEXT_ID      = os.environ.get("CLIP_MODEL_ID_TXT", "sentence-transformers/clip-ViT-B-32-multilingual-v1")
BLIP2_MODEL_ID    = os.environ.get("BLIP2_MODEL_ID", "Salesforce/blip2-flan-t5-xl")

INIT_BATCH = int(os.environ.get("BATCH_SIZE", "32"))
TXT_BATCH  = int(os.environ.get("TXT_BATCH",  "32"))
MAX_ITEMS  = os.environ.get("MAX_ITEMS", None)
MAX_ITEMS  = None if MAX_ITEMS in [None, "", "None"] else int(MAX_ITEMS)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
torch.set_float32_matmul_precision("high")

EPS = 1e-6
STATS_CHUNK = 8192   # rows per chunk when computing mu/sigma and applying z-score

# ------------- Utilities -------------
def list_images(root: Path) -> List[Path]:
    imgs = list(root.glob("*.jpg")) + list(root.glob("*.JPG")) \
         + list(root.glob("*.jpeg")) + list(root.glob("*.JPEG"))
    imgs.sort()
    return imgs

def list_jsons(root: Path) -> List[Path]:
    files = list(root.glob("*.json"))
    files.sort()
    return files

def load_image_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")

def free_cuda(label=""):
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()

def create_memmap(path: Path, shape, dtype=np.float32):
    from numpy.lib.format import open_memmap
    return open_memmap(str(path), mode='w+', dtype=dtype, shape=shape)

def compute_stats_mm(mm: np.memmap) -> Tuple[np.ndarray, np.ndarray]:
    """Two-pass (sum/sumsq) numerically-stable per-dim stats over a (N, D) memmap."""
    N, D = mm.shape
    s = np.zeros(D, dtype=np.float64)
    s2 = np.zeros(D, dtype=np.float64)

    for i in range(0, N, STATS_CHUNK):
        sl = slice(i, min(i + STATS_CHUNK, N))
        x = mm[sl].astype(np.float64, copy=False)
        s  += x.sum(axis=0)
        s2 += (x * x).sum(axis=0)

    mu = s / max(N, 1)
    var = np.maximum(s2 / max(N, 1) - mu * mu, 0.0)
    sigma = np.sqrt(var)
    return mu.astype(np.float32), sigma.astype(np.float32)

def apply_zscore_inplace(mm: np.memmap, mu: np.ndarray, sigma: np.ndarray):
    N = mm.shape[0]
    den = (sigma + EPS).astype(np.float32, copy=False)
    for i in range(0, N, STATS_CHUNK):
        sl = slice(i, min(i + STATS_CHUNK, N))
        mm[sl] = (mm[sl] - mu) / den

# ------------------------------------
# CLIP (image) → single pooled vector
# ------------------------------------
def dump_clip_image_pooled(out_path: Path, stats_path: Path):
    from transformers import CLIPProcessor, CLIPModel

    print(f"🔎 Scanning posters in: {DATA_DIR}")
    images = list_images(DATA_DIR)
    if MAX_ITEMS is not None: images = images[:MAX_ITEMS]
    if not images:
        print("❌ No images found."); sys.exit(1)
    print(f"🖼️  Found {len(images)} images")

    print(f"🧠 Loading CLIP image: {CLIP_MODEL_ID_IMG}")
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID_IMG)
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID_IMG,
                                      torch_dtype=DTYPE,
                                      low_cpu_mem_usage=True).to(DEVICE).eval()

    # Probe dim
    with torch.inference_mode():
        dummy = Image.new("RGB", (224, 224), (0,0,0))
        toks = processor(images=dummy, return_tensors="pt")
        toks = {k: v.to(DEVICE) for k, v in toks.items()}
        feats = model.get_image_features(**toks)    # [1, 512]
        D = feats.shape[-1]
    print(f"📐 CLIP image embed dim = {D}")

    N = len(images)
    mm = create_memmap(out_path, (N, D), dtype=np.float32)

    bs = max(1, min(INIT_BATCH, 64))
    pbar, i = tqdm(total=N, desc="CLIP(image) pooled", unit="img"), 0

    while i < N:
        j = min(i + bs, N)
        batch_imgs, idxs = [], []
        for k in range(i, j):
            p = images[k]
            try:
                batch_imgs.append(load_image_rgb(p))
                idxs.append(k)
            except (UnidentifiedImageError, OSError) as e:
                print(f"⚠️  Skipping unreadable image: {p.name} ({e})")
        if not batch_imgs: i = j; continue

        try:
            with torch.inference_mode():
                toks = processor(images=batch_imgs, return_tensors="pt")
                toks = {k: v.to(DEVICE) for k, v in toks.items()}
                vecs = model.get_image_features(**toks)   # [B, 512]
                vecs = vecs.float().cpu().numpy()
            for row, k in enumerate(idxs): mm[k] = vecs[row]
            pbar.update(len(idxs)); i = j
            if bs < 128: bs = min(128, bs * 2)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                free_cuda("clip_image OOM")
                bs = max(1, bs // 2)
                print(f"💥 OOM — reducing image batch_size to {bs} …")
            else:
                print("❌ Runtime error:", e); traceback.print_exc(); i = j

    pbar.close()
    del model, processor; free_cuda("clip_image done")

    # Z-score across dataset
    print("📊 Z-scoring CLIP(image) …")
    mu, sigma = compute_stats_mm(mm)
    apply_zscore_inplace(mm, mu, sigma)
    np.savez_compressed(stats_path, mu=mu, sigma=sigma)
    print(f"💾 Saved: {out_path} {mm.shape} and stats: {stats_path}")

# -----------------------------------
# CLIP (text) → single pooled vector
# -----------------------------------
def dump_clip_text_pooled(out_path: Path, stats_path: Path):
    from sentence_transformers import SentenceTransformer

    print(f"🔎 Scanning JSONs in {DATA_DIR}")
    files = list_jsons(DATA_DIR)
    if MAX_ITEMS is not None: files = files[:MAX_ITEMS]
    if not files:
        print("❌ No JSON files found."); sys.exit(1)
    print(f"📄 Found {len(files)} JSON files")

    print(f"🧠 Loading Sentence-Transformers: {CLIP_TEXT_ID}")
    st = SentenceTransformer(CLIP_TEXT_ID, device=str(DEVICE))
    st = st.eval()  # keep float32 to avoid dtype quirks

    # Probe D
    with torch.inference_mode():
        probe = st.encode(["hello world"], normalize_embeddings=False, convert_to_numpy=True)
        D = probe.shape[-1]
    print(f"📐 CLIP text embed dim = {D}")

    N = len(files)
    mm = create_memmap(out_path, (N, D), dtype=np.float32)

    bs = max(1, min(TXT_BATCH, 128))
    pbar, i = tqdm(total=N, desc="CLIP(text) pooled", unit="synopsis"), 0

    while i < N:
        j = min(i + bs, N)
        texts, idxs = [], []
        for k in range(i, j):
            p = files[k]
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                texts.append(data.get("plot", "") or "")
                idxs.append(k)
            except Exception as e:
                print(f"⚠️  Skipping {p.name}: {e}")

        if not texts: i = j; continue

        try:
            vecs = st.encode(texts,
                             normalize_embeddings=False,
                             convert_to_numpy=True,
                             batch_size=bs,
                             show_progress_bar=False)
            for row, k in enumerate(idxs): mm[k] = vecs[row].astype(np.float32, copy=False)
            pbar.update(len(idxs)); i = j
            if bs < 256: bs = min(256, bs * 2)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                free_cuda("clip_text OOM")
                bs = max(1, bs // 2)
                print(f"💥 OOM — reducing text batch_size to {bs} …")
            else:
                print("❌ Runtime error:", e); traceback.print_exc(); i = j

    pbar.close()
    del st; free_cuda("clip_text done")

    print("📊 Z-scoring CLIP(text) …")
    mu, sigma = compute_stats_mm(mm)
    apply_zscore_inplace(mm, mu, sigma)
    np.savez_compressed(stats_path, mu=mu, sigma=sigma)
    print(f"💾 Saved: {out_path} {mm.shape} and stats: {stats_path}")

# -----------------------------------
# BLIP-2 (image) → pooled Q-Former
# -----------------------------------
def dump_blip2_image_pooled(out_path: Path, stats_path: Path):
    from transformers import Blip2Processor, Blip2Model

    print(f"🔎 Scanning posters in: {DATA_DIR}")
    images = list_images(DATA_DIR)
    if MAX_ITEMS is not None: images = images[:MAX_ITEMS]
    if not images:
        print("❌ No images found."); sys.exit(1)
    print(f"🖼️  Found {len(images)} images")

    print(f"🧠 Loading BLIP-2: {BLIP2_MODEL_ID}")
    processor = Blip2Processor.from_pretrained(BLIP2_MODEL_ID)
    model = Blip2Model.from_pretrained(BLIP2_MODEL_ID,
                                       torch_dtype=DTYPE,
                                       low_cpu_mem_usage=True).to(DEVICE).eval()

    # Probe D (Q-Former hidden)
    with torch.inference_mode():
        dummy = Image.new("RGB", (224, 224), (0, 0, 0))
        toks = processor(images=dummy, return_tensors="pt")
        toks = {k: v.to(DEVICE) for k, v in toks.items()}
        if hasattr(model, "get_qformer_features"):
            q = model.get_qformer_features(**toks).last_hidden_state   # [1, Q, 768]
        else:
            q = model(pixel_values=toks["pixel_values"], return_dict=True).qformer_last_hidden_state
        D = q.shape[-1]
    print(f"📐 BLIP-2 image pooled dim = {D} (Q-Former → mean over Q)")

    N = len(images)
    mm = create_memmap(out_path, (N, D), dtype=np.float32)

    bs = max(1, min(INIT_BATCH // 8, 4))  # very conservative start
    pbar, i = tqdm(total=N, desc="BLIP-2(image) pooled", unit="img"), 0

    while i < N:
        j = min(i + bs, N)
        batch_imgs, idxs = [], []
        for k in range(i, j):
            p = images[k]
            try:
                batch_imgs.append(load_image_rgb(p))
                idxs.append(k)
            except (UnidentifiedImageError, OSError) as e:
                print(f"⚠️  Skipping unreadable image: {p.name} ({e})")
        if not batch_imgs: i = j; continue

        try:
            with torch.inference_mode():
                toks = processor(images=batch_imgs, return_tensors="pt")
                toks = {k: v.to(DEVICE) for k, v in toks.items()}
                if hasattr(model, "get_qformer_features"):
                    q = model.get_qformer_features(**toks).last_hidden_state  # [B, Q, 768]
                else:
                    q = model(pixel_values=toks["pixel_values"], return_dict=True).qformer_last_hidden_state
                vecs = q.mean(dim=1)  # [B, 768]
                vecs = vecs.float().cpu().numpy()
            for row, k in enumerate(idxs): mm[k] = vecs[row]
            pbar.update(len(idxs)); i = j
            if bs < 8: bs = min(8, bs * 2)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                free_cuda("blip2_image OOM")
                bs = max(1, bs // 2)
                print(f"💥 OOM — reducing blip2-image batch_size to {bs} …")
            else:
                print("❌ Runtime error:", e); traceback.print_exc(); i = j

    pbar.close()
    del model, processor; free_cuda("blip2_image done")

    print("📊 Z-scoring BLIP-2(image) …")
    mu, sigma = compute_stats_mm(mm)
    apply_zscore_inplace(mm, mu, sigma)
    np.savez_compressed(stats_path, mu=mu, sigma=sigma)
    print(f"💾 Saved: {out_path} {mm.shape} and stats: {stats_path}")

# -----------------------------------
# BLIP-2 (text) → pooled LM encoder
# -----------------------------------
def dump_blip2_text_pooled(out_path: Path, stats_path: Path):
    from transformers import Blip2Processor, Blip2Model

    print(f"🔎 Scanning JSONs in {DATA_DIR}")
    files = list_jsons(DATA_DIR)
    if MAX_ITEMS is not None: files = files[:MAX_ITEMS]
    if not files:
        print("❌ No JSON files found."); sys.exit(1)
    print(f"📄 Found {len(files)} JSON files")

    print(f"🧠 Loading BLIP-2: {BLIP2_MODEL_ID}")
    processor = Blip2Processor.from_pretrained(BLIP2_MODEL_ID)
    model = Blip2Model.from_pretrained(BLIP2_MODEL_ID,
                                       torch_dtype=DTYPE,
                                       low_cpu_mem_usage=True).to(DEVICE).eval()

    def lm_last_hidden(tok: dict) -> torch.Tensor:
        lm = model.language_model
        cls = lm.__class__.__name__.lower()
        with torch.inference_mode():
            if "t5" in cls:
                enc = lm.encoder(input_ids=tok["input_ids"],
                                 attention_mask=tok["attention_mask"],
                                 output_hidden_states=True, return_dict=True)
                return enc.last_hidden_state     # [B, T, H]
            else:
                base = getattr(lm, "model", lm)  # OPT exposes .model
                out  = base(input_ids=tok["input_ids"],
                            attention_mask=tok["attention_mask"],
                            output_hidden_states=True, return_dict=True)
                return out.last_hidden_state     # [B, T, H]

    # Probe dims
    with torch.inference_mode():
        tp = processor.tokenizer(["hello world"], padding=True, truncation=True,
                                 max_length=512, return_tensors="pt")
        tp = {k: v.to(DEVICE) for k, v in tp.items()}
        last = lm_last_hidden(tp)    # [1, T, H]
        D = last.shape[-1]
        MAX_LEN = tp["input_ids"].shape[1]
    print(f"📐 BLIP-2 text pooled dim = {D} (mean over non-pad tokens, max_len={MAX_LEN})")

    N = len(files)
    mm = create_memmap(out_path, (N, D), dtype=np.float32)

    bs = max(1, min(TXT_BATCH // 2, 8))
    pbar, i = tqdm(total=N, desc="BLIP-2(text) pooled", unit="synopsis"), 0

    while i < N:
        j = min(i + bs, N)
        texts, idxs = [], []
        for k in range(i, j):
            p = files[k]
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                texts.append(data.get("plot", "") or "")
                idxs.append(k)
            except Exception as e:
                print(f"⚠️  Skipping {p.name}: {e}")
        if not texts: i = j; continue

        try:
            tok = processor.tokenizer(texts, padding=True, truncation=True,
                                      max_length=MAX_LEN, return_tensors="pt")
            tok = {k: v.to(DEVICE) for k, v in tok.items()}
            with torch.inference_mode():
                last = lm_last_hidden(tok)               # [B, T, H]
                mask = tok["attention_mask"].unsqueeze(-1).to(last.dtype)  # [B, T, 1]
                denom = torch.clamp(mask.sum(dim=1), min=1.0)              # [B, 1]
                vecs = (last * mask).sum(dim=1) / denom                    # [B, H]
                vecs = vecs.float().cpu().numpy()

            for row, k in enumerate(idxs): mm[k] = vecs[row]
            pbar.update(len(idxs)); i = j
            if bs < 16: bs = min(16, bs * 2)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                free_cuda("blip2_text OOM")
                bs = max(1, bs // 2)
                print(f"💥 OOM — reducing blip2-text batch_size to {bs} …")
            else:
                print("❌ Runtime error:", e); traceback.print_exc(); i = j

    pbar.close()
    del model, processor; free_cuda("blip2_text done")

    print("📊 Z-scoring BLIP-2(text) …")
    mu, sigma = compute_stats_mm(mm)
    apply_zscore_inplace(mm, mu, sigma)
    np.savez_compressed(stats_path, mu=mu, sigma=sigma)
    print(f"💾 Saved: {out_path} {mm.shape} and stats: {stats_path}")

# ------------- Main CLI -------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="all",
        choices=["clip_image", "clip_text", "blip2_image", "blip2_text", "all"],
        help="Which dump(s) to run; default: all")
    parser.add_argument("--out", type=str, default=None,
        help="Output path for single-mode run (.npy enforced).")
    parser.add_argument("--out-prefix", type=str, default=None,
        help="If --mode all, write <prefix>_{clip_image.npy,clip_text.npy,blip2_image.npy,blip2_text.npy}")
    args = parser.parse_args()

    print(f"📂 DATA_DIR = {DATA_DIR}")
    print(f"🧮 DTYPE={DTYPE}  DEVICE={DEVICE}")

    if args.mode == "all":
        if args.out and not args.out_prefix:
            print("⚠️  --out is ignored in --mode all. Use --out-prefix.")
        prefix = args.out_prefix or "mm_imdb2_globals"

        out_clip_img  = Path(f"{prefix}_clip_image.npy")
        out_clip_txt  = Path(f"{prefix}_clip_text.npy")
        out_blip2_img = Path(f"{prefix}_blip2_image.npy")
        out_blip2_txt = Path(f"{prefix}_blip2_text.npy")

        stats_clip_img  = Path(f"{prefix}_clip_image.stats.npz")
        stats_clip_txt  = Path(f"{prefix}_clip_text.stats.npz")
        stats_blip2_img = Path(f"{prefix}_blip2_image.stats.npz")
        stats_blip2_txt = Path(f"{prefix}_blip2_text.stats.npz")

        print(f"🚀 Running ALL dumps → prefix={prefix}")
        dump_clip_image_pooled(out_clip_img, stats_clip_img)
        dump_clip_text_pooled(out_clip_txt, stats_clip_txt)
        dump_blip2_image_pooled(out_blip2_img, stats_blip2_img)
        dump_blip2_text_pooled(out_blip2_txt, stats_blip2_txt)
        return

    # Single-mode path
    mode = args.mode
    if args.out:
        out_path = Path(args.out)
        if out_path.suffix.lower() != ".npy":
            out_path = out_path.with_suffix(".npy")
        prefix = out_path.with_suffix("")  # for stats path
    else:
        defaults = {
            "clip_image": "mm_imdb2_globals_clip_image.npy",
            "clip_text":  "mm_imdb2_globals_clip_text.npy",
            "blip2_image":"mm_imdb2_globals_blip2_image.npy",
            "blip2_text": "mm_imdb2_globals_blip2_text.npy",
        }
        out_path = Path(defaults[mode])
        prefix = out_path.with_suffix("")

    stats_path = Path(str(prefix) + ".stats.npz")

    print(f"🚀 Mode: {mode} → {out_path}")
    if mode == "clip_image":
        dump_clip_image_pooled(out_path, stats_path)
    elif mode == "clip_text":
        dump_clip_text_pooled(out_path, stats_path)
    elif mode == "blip2_image":
        dump_blip2_image_pooled(out_path, stats_path)
    elif mode == "blip2_text":
        dump_blip2_text_pooled(out_path, stats_path)

if __name__ == "__main__":
    main()
