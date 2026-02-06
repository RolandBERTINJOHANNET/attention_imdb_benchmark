Reproduce the mmm-imdb benchmark results from the paper "An Attention Mechanism for Robust Multimodal Integration in a Global Workspace Architecture"

## Install
- Python 3.11
- `poetry install`

## Data expected (IMDB1 aug-70 latents)
You can generate the four embedding `.npy` files with the provided scripts; set paths via `example.env`.
- Raw IMDB1 root (with `split.json`): `MMIMDB1_DIR`
- Output dir for embeddings: `EMB_OUT_DIR`
- Labels dir (contains `labels_all_23.npy`, `ids_all.txt`): `IMDB1_LABELS_DIR_23`
Defaults are in `imdb_gw/data/dataset_1x_aug70.py`; override via env vars or `example.env`.

## Run the command to replicate the numbers
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
