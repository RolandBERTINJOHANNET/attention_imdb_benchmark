#!/usr/bin/env python3
# make_mm_imdb1_clip_blip2_embeddings_aug70_batched.py
# ---------------------------------------------------------------------
# Same as your aug70 script, but FAST:
#   • Builds all 70 views for a mini-batch of original samples,
#   • Encodes per modality in large batches (chunked),
#   • Preserves determinism and global index alignment.
# ---------------------------------------------------------------------

import os, sys, gc, json, traceback, time, random, hashlib
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

# ---------------- Config (env-overridable) ----------------
DATA_DIR   = Path(os.getenv("MMIMDB1_DATASET_DIR", "./data/imdb1/dataset"))
OUT_DIR    = Path(os.getenv("EMB_OUT_DIR", "./data/embeddings_clip_blip2_aug70"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PREFIX = "mm_imdb1_globals_aug70"

CLIP_MODEL_ID_IMG = "openai/clip-vit-base-patch32"
CLIP_TEXT_ID      = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
BLIP2_MODEL_ID    = "Salesforce/blip2-flan-t5-xl"

# Views / batching
VIEWS          = 70          # 0=clean, 1..69=aug
ORIG_BATCH     = 256          # originals per step (↑ this to improve throughput)
CLIP_IMG_BS    = 256         # micro-batch for CLIP image
CLIP_TXT_BS    = 256         # micro-batch for ST text
BLIP_IMG_BS    = 16          # micro-batch for BLIP-2 image
BLIP_TXT_BS    = 16          # micro-batch for BLIP-2 text

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    DTYPE = torch.float32
torch.set_float32_matmul_precision("high")

EPS = 1e-6
STATS_CHUNK = 8192
BASE_SEED = 123

# ---------------- Utils ----------------
def free_cuda(_=""):
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass
    import gc; gc.collect()

def create_memmap(path: Path, shape, dtype=np.float32):
    from numpy.lib.format import open_memmap
    return open_memmap(str(path), mode='w+', dtype=dtype, shape=shape)

def compute_stats_mm(mm: np.memmap):
    N, D = mm.shape
    s = np.zeros(D, np.float64); s2 = np.zeros(D, np.float64)
    for i in range(0, N, STATS_CHUNK):
        sl = slice(i, min(i + STATS_CHUNK, N))
        x = mm[sl].astype(np.float64, copy=False)
        s  += x.sum(0); s2 += (x * x).sum(0)
    mu = s / max(N, 1); var = np.maximum(s2 / max(N, 1) - mu * mu, 0.0)
    return mu.astype(np.float32), np.sqrt(var).astype(np.float32)

def apply_zscore_inplace(mm: np.memmap, mu: np.ndarray, sigma: np.ndarray):
    den = (sigma + EPS).astype(np.float32, copy=False)
    for i in range(0, mm.shape[0], STATS_CHUNK):
        sl = slice(i, min(i + STATS_CHUNK, mm.shape[0]))
        mm[sl] = (mm[sl] - mu) / den

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

def load_image_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")

def read_plot_text(js_path: Path) -> str:
    with open(js_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    t = d.get("plot", "") or ""
    return " ".join(x for x in t if isinstance(t, list)) if isinstance(t, list) else t

# ---- SimCLR augs (deterministic per (orig_idx, view)) ----
from typing import Tuple  # noqa
def _simclr_kernel(image_size: int = 224) -> int:
    k = max(3, int(0.1 * image_size));  k += (k % 2 == 0)
    return k

def build_simclr_transform(image_size: int = 224) -> T.Compose:
    k = _simclr_kernel(image_size)
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=k, sigma=(0.1, 2.0))], p=0.5),
    ])

def seed_for(orig_idx: int, view: int, salt: str) -> int:
    h = hashlib.sha256(f"{BASE_SEED}:{orig_idx}:{view}:{salt}".encode()).hexdigest()
    return int(h[:8], 16)

def apply_simclr_det(pil: Image.Image, tfm: T.Compose, orig_idx: int, view: int) -> Image.Image:
    s = seed_for(orig_idx, view, "image")
    random.seed(s); np.random.seed(s & 0x7fffffff)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    return tfm(pil)

# ---- BERT-style text masking (15%) (deterministic) ----
def mask_text_bert_style(text: str, orig_idx: int, view: int, mask_prob: float = 0.15) -> str:
    if view == 0 or not text: return text
    rng = random.Random(seed_for(orig_idx, view, "text"))
    toks = text.split(); n = len(toks)
    if n == 0: return text
    k = max(1, int(round(mask_prob * n)))
    idxs = list(range(n)); rng.shuffle(idxs)
    for j in idxs[:k]: toks[j] = "[MASK]"
    return " ".join(toks)

# ---------------- Encoders (exact procedures) ----------------
def init_clip_image():
    from transformers import CLIPProcessor, CLIPModel
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL_ID_IMG)
    model = CLIPModel.from_pretrained(
        CLIP_MODEL_ID_IMG,
        torch_dtype=DTYPE if DEVICE.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    ).to(DEVICE).eval()
    with torch.inference_mode():
        dummy = Image.new("RGB", (224,224), (0,0,0))
        tok = proc(images=dummy, return_tensors="pt")
        tok = {k: v.to(DEVICE) for k, v in tok.items()}
        D = model.get_image_features(**tok).shape[-1]
    return proc, model, D

def init_clip_text():
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(CLIP_TEXT_ID, device=str(DEVICE)).eval()
    with torch.inference_mode():
        D = st.encode(["hello"], normalize_embeddings=False, convert_to_numpy=True).shape[-1]
    return st, D

def init_blip2():
    from transformers import Blip2Processor, Blip2Model
    proc = Blip2Processor.from_pretrained(BLIP2_MODEL_ID)
    model = Blip2Model.from_pretrained(
        BLIP2_MODEL_ID,
        torch_dtype=DTYPE if DEVICE.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    ).to(DEVICE).eval()
    with torch.inference_mode():
        # Q-Former dim
        dummy = Image.new("RGB", (224,224), (0,0,0))
        ti = proc(images=dummy, return_tensors="pt")
        ti = {k: v.to(DEVICE) for k, v in ti.items()}
        try:
            q = model.get_qformer_features(**ti).last_hidden_state
        except AttributeError:
            q = model(pixel_values=ti["pixel_values"], return_dict=True).qformer_last_hidden_state
        D_img = q.shape[-1]
        # LM enc dim (+ max len)
        tt = proc.tokenizer(["hello world"], padding=True, truncation=True, max_length=512, return_tensors="pt")
        tt = {k: v.to(DEVICE) for k, v in tt.items()}
        lm = model.language_model; cls = lm.__class__.__name__.lower()
        if "t5" in cls:
            enc = lm.encoder(input_ids=tt["input_ids"], attention_mask=tt["attention_mask"],
                             output_hidden_states=True, return_dict=True)
            last = enc.last_hidden_state
        else:
            base = getattr(lm, "model", lm)
            out  = base(input_ids=tt["input_ids"], attention_mask=tt["attention_mask"],
                        output_hidden_states=True, return_dict=True)
            last = out.last_hidden_state
        D_txt = last.shape[-1]; MAX_LEN = tt["input_ids"].shape[1]
    return proc, model, D_img, D_txt, MAX_LEN

# ---------------- Main (batched across views) ----------------
def main():
    t0 = time.time()
    print("=== MM-IMDb 1.0 → CLIP/BLIP2 pooled embeddings with 70 aligned views (BATCHED) ===")
    print(f"📂 DATA_DIR={DATA_DIR}\n🗃️ OUT_DIR={OUT_DIR}\n🧮 DEVICE={DEVICE} DTYPE={DTYPE} VIEWS={VIEWS}")

    pairs = pair_samples_by_stem(DATA_DIR); N = len(pairs)
    print(f"🔗 Aligned pairs: {N}")
    texts = [read_plot_text(js) for (_, js) in pairs]

    # Encoders
    clip_proc, clip_model, D_clip_img = init_clip_image()
    st_model, D_clip_txt = init_clip_text()
    blip_proc, blip_model, D_blip_img, D_blip_txt, MAX_LEN = init_blip2()

    simclr = build_simclr_transform(224)

    total_rows = N * VIEWS
    p_ci, st_ci = out_paths("clip_image")
    p_ct, st_ct = out_paths("clip_text")
    p_bi, st_bi = out_paths("blip2_image")
    p_bt, st_bt = out_paths("blip2_text")

    mm_ci = create_memmap(p_ci, (total_rows, D_clip_img), np.float32)
    mm_ct = create_memmap(p_ct, (total_rows, D_clip_txt), np.float32)
    mm_bi = create_memmap(p_bi, (total_rows, D_blip_img), np.float32)
    mm_bt = create_memmap(p_bt, (total_rows, D_blip_txt), np.float32)

    index_csv = OUT_DIR / f"{OUT_PREFIX}_aug_index.csv"
    with index_csv.open("w", encoding="utf-8") as f:
        f.write("aug_idx,orig_idx,view,stem,image_name,text_len\n")

    prog = tqdm(total=N, desc="Aug70 (batched originals)", unit="orig")
    for start in range(0, N, ORIG_BATCH):
        end = min(start + ORIG_BATCH, N); B = end - start
        batch_pairs = pairs[start:end]
        batch_texts = texts[start:end]
        stems = [p[0].stem for p in batch_pairs]

        # Load clean PILs
        pil_clean: List[Image.Image] = []
        for (img_path, _) in batch_pairs:
            try:
                pil_clean.append(load_image_rgb(img_path))
            except (UnidentifiedImageError, OSError):
                pil_clean.append(Image.new("RGB", (224,224), (0,0,0)))

        # ---------- Build ALL 70 views first (lists length B*V) ----------
        imgs_all: List[Image.Image] = []
        txts_all: List[str] = []
        gidx_all: List[int] = []
        meta_lines: List[str] = []

        for v in range(VIEWS):
            for bi in range(B):
                orig_idx = start + bi
                gidx = orig_idx * VIEWS + v
                if v == 0:
                    imgv = pil_clean[bi]
                    txtv = batch_texts[bi]
                else:
                    imgv = apply_simclr_det(pil_clean[bi], simclr, orig_idx, v)
                    txtv = mask_text_bert_style(batch_texts[bi], orig_idx, v, 0.15)
                imgs_all.append(imgv)
                txts_all.append(txtv)
                gidx_all.append(gidx)
                meta_lines.append(f"{gidx},{orig_idx},{v},{stems[bi]},{batch_pairs[bi][0].name},{len(txtv.split())}\n")

        # ---------- Encode per modality in chunks ----------
        # CLIP image
        for i0 in range(0, len(imgs_all), CLIP_IMG_BS):
            i1 = min(i0 + CLIP_IMG_BS, len(imgs_all))
            with torch.inference_mode():
                tok = clip_proc(images=imgs_all[i0:i1], return_tensors="pt")
                tok = {k: v.to(DEVICE) for k, v in tok.items()}
                vec = clip_model.get_image_features(**tok).float().cpu().numpy()
            for r, g in enumerate(gidx_all[i0:i1]):
                mm_ci[g] = vec[r]

        # CLIP text
        with torch.inference_mode():
            vec_txt = st_model.encode(txts_all, batch_size=CLIP_TXT_BS,
                                      normalize_embeddings=False, convert_to_numpy=True,
                                      show_progress_bar=False)
        if isinstance(vec_txt, np.ndarray):
            for r, g in enumerate(gidx_all):
                mm_ct[g] = vec_txt[r].astype(np.float32, copy=False)
        else:
            arr = np.stack([t.cpu().numpy() for t in vec_txt], 0).astype(np.float32, copy=False)
            for r, g in enumerate(gidx_all):
                mm_ct[g] = arr[r]

        # BLIP-2 image (Q-Former mean)
        for i0 in range(0, len(imgs_all), BLIP_IMG_BS):
            i1 = min(i0 + BLIP_IMG_BS, len(imgs_all))
            with torch.inference_mode():
                tok = blip_proc(images=imgs_all[i0:i1], return_tensors="pt")
                tok = {k: v.to(DEVICE) for k, v in tok.items()}
                try:
                    q = blip_model.get_qformer_features(**tok).last_hidden_state
                except AttributeError:
                    q = blip_model(pixel_values=tok["pixel_values"], return_dict=True).qformer_last_hidden_state
                vec = q.mean(1).float().cpu().numpy()
            for r, g in enumerate(gidx_all[i0:i1]):
                mm_bi[g] = vec[r]

        # BLIP-2 text (LM encoder mean over non-pad)
        for i0 in range(0, len(txts_all), BLIP_TXT_BS):
            i1 = min(i0 + BLIP_TXT_BS, len(txts_all))
            sub = txts_all[i0:i1]
            tok = blip_proc.tokenizer(sub, padding=True, truncation=True,
                                      max_length=512, return_tensors="pt")
            tok = {k: v.to(DEVICE) for k, v in tok.items()}
            with torch.inference_mode():
                lm = blip_model.language_model; cls = lm.__class__.__name__.lower()
                if "t5" in cls:
                    enc = lm.encoder(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"],
                                     output_hidden_states=True, return_dict=True)
                    last = enc.last_hidden_state
                else:
                    base = getattr(lm, "model", lm)
                    out  = base(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"],
                                output_hidden_states=True, return_dict=True)
                    last = out.last_hidden_state
                mask = tok["attention_mask"].unsqueeze(-1).to(last.dtype)
                denom = torch.clamp(mask.sum(1), min=1.0)
                vec = (last * mask).sum(1) / denom
                vec = vec.float().cpu().numpy()
            for r, g in enumerate(gidx_all[i0:i1]):
                mm_bt[g] = vec[r]

        # Write index metadata once per orig-batch
        with index_csv.open("a", encoding="utf-8") as f:
            f.writelines(meta_lines)

        prog.update(B); free_cuda("batch done")
    prog.close()

    # -------- Z-score files --------
    for (mm, stats, label) in [
        (mm_ci, st_ci, "CLIP(image)"),
        (mm_ct, st_ct, "CLIP(text)"),
        (mm_bi, st_bi, "BLIP2(image)"),
        (mm_bt, st_bt, "BLIP2(text)"),
    ]:
        print(f"📊 Z-scoring {label} …")
        mu, sigma = compute_stats_mm(mm)
        apply_zscore_inplace(mm, mu, sigma)
        np.savez_compressed(stats, mu=mu, sigma=sigma)
        print(f"💾 Saved stats → {stats}")

    # Manifest
    manifest = {
        "total_rows": int(N * VIEWS),
        "original_rows": int(N),
        "views_per_sample": int(VIEWS),
        "dims": {"clip_image": int(mm_ci.shape[1]), "clip_text": int(mm_ct.shape[1]),
                 "blip2_image": int(mm_bi.shape[1]), "blip2_text": int(mm_bt.shape[1])},
        "models": {"clip_image": CLIP_MODEL_ID_IMG, "clip_text": CLIP_TEXT_ID, "blip2": BLIP2_MODEL_ID},
        "dtype": str(DTYPE), "device": str(DEVICE),
        "simclr": True, "text_mask_prob": 0.15,
        "index_csv": str(index_csv),
    }
    (OUT_DIR / f"{OUT_PREFIX}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("✅ Done.")
    print(f"• {p_ci.name} shape={mm_ci.shape}\n• {p_ct.name} shape={mm_ct.shape}\n• {p_bi.name} shape={mm_bi.shape}\n• {p_bt.name} shape={mm_bt.shape}\n• Index CSV: {index_csv}")
    print(f"⏱️  Elapsed: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print("❌ Unexpected error:", e)
        traceback.print_exc()
        sys.exit(99)
