# imdb_benchmark_gw_attention

Minimal code to reproduce the ATT-GW orchestrator run on IMDB1 aug-70 latents.

## Install
- Python 3.11
- `poetry install`

## Data expected (IMDB1 aug-70 latents)
You can generate the four embedding `.npy` files with the provided scripts; set paths via `example.env`.
- Raw IMDB1 root (with `split.json`): `MMIMDB1_DIR`
- Output dir for embeddings: `EMB_OUT_DIR`
- Labels dir (contains `labels_all_23.npy`, `ids_all.txt`): `IMDB1_LABELS_DIR_23`
Defaults are in `imdb_gw/data/dataset_1x_aug70.py`; override via env vars or `example.env`.

## Run the target command
```bash
poetry run orchestrate-attgw
# or
poetry run python - <<'PY'
import imdb_gw.training.final_orchestrator as O
O.orchestrate_attgw(
    backbones=("blip2",),
    selectors=("shared","random"),
    seeds=(42,),
    results_root="./attgw_results",
    random_temperature=0.1,
)
PY
```

## Included code
- Orchestrator: `imdb_gw/training/final_orchestrator.py`
- Train one run: `imdb_gw/training/function_train_attgw.py`
- Eval: `imdb_gw/training/function_eval_attgw.py`
- Data prep: `imdb_gw/data/dataset_1x_aug70.py`
- Optional generators: `make_mm_imdb1_clip_blip2_embeddings.py`, `make_mm_imdb2_clip_blip2_embeddings.py`
- Label prep: `augment_labels.py`
- Domains: `imdb_gw/domains/domains_obb.py`
- Shimmer monkeypatches: `imdb_gw/overrides/shimmer_patches.py`

Notes: CUDA required; WandB removed (CSV/ckpt logging only).
