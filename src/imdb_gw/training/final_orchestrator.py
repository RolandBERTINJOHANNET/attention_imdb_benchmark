#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
orchestrate_attgw.py
--------------------
Trains GW models with selectors {"shared","random"} on backbones {"clip","blip2"}
for seeds {42, 1337, 2025}, evaluates on CLEAN, and saves JSON/CSV results into
a unique, non-overwriting folder.

Imports:
  - instantiate_and_train from function_train_attgw.py
  - evaluate_clean        from function_eval_attgw.py
"""

from __future__ import annotations
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List
from zoneinfo import ZoneInfo

import torch

from imdb_gw.training.function_train_attgw import instantiate_and_train
from imdb_gw.training.function_eval_attgw import evaluate_clean


# ---------- utils ----------
def _now_paris() -> datetime:
    return datetime.now(ZoneInfo("Europe/Paris"))

def _make_safe_outdir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    ts = _now_paris().strftime("%Y-%m-%d_%H-%M-%S")
    candidate = base / f"run_{ts}"
    i = 2
    while candidate.exists():
        candidate = base / f"run_{ts}_{i}"
        i += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate

@dataclass
class RunSpec:
    backbone: str
    selector: str   # "shared" | "random"
    seed: int

def _device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------- orchestrator ----------
def orchestrate_attgw(
    backbones: Iterable[str] = ("blip2", "clip"),
    selectors: Iterable[str] = ("shared", "random"),
    seeds: Iterable[int] = (42, 1337, 2025),
    results_root: str | Path = "./attgw_results",
    random_temperature: float = 0.1,  # lower -> peakier random softmax
) -> Dict:
    artifacts_dir = _make_safe_outdir(Path(results_root))
    json_path = artifacts_dir / "results.json"
    csv_path  = artifacts_dir / "results.csv"
    meta_path = artifacts_dir / "meta.json"

    grid: List[RunSpec] = [RunSpec(b, s, int(seed)) for b in backbones for s in selectors for seed in seeds]

    rows: List[Dict] = []
    counts = {"OK": 0, "FAIL_TRAIN": 0, "FAIL_EVAL": 0}

    print(f"[ATTGW] Orchestrating {len(grid)} runs on device={_device_str()}")
    print(f"[ATTGW] Artifacts dir: {artifacts_dir}\n")

    for i, spec in enumerate(grid, 1):
        if spec.selector not in {"shared", "random"}:
            print(f"[{i}/{len(grid)}] ❌ INVALID SELECTOR {spec.selector} (expected 'shared' or 'random')")
            rows.append({"status":"FAIL_TRAIN","backbone":spec.backbone,"selector":spec.selector,"seed":spec.seed,
                         "error":"Invalid selector","train_seconds":None,"eval_seconds":None,"device":_device_str()})
            counts["FAIL_TRAIN"] += 1
            continue

        tag = f"{spec.backbone} | selector={spec.selector:<6} | seed={spec.seed}"
        if spec.selector == "random":
            tag += f" | temp={random_temperature:g}"
        print(f"[{i}/{len(grid)}] ▶️  TRAIN  {tag}")
        t0 = time.perf_counter()
        try:
            model = instantiate_and_train(
                seed=spec.seed,
                backbone=spec.backbone,
                selector=spec.selector,                # "shared" or "random"
                temperature=random_temperature,        # <- make random attention spikier/flatter
            )
            t_train = time.perf_counter() - t0
        except Exception as e:
            print(f"[{i}/{len(grid)}] ❌ TRAIN FAILED {tag} :: {type(e).__name__}: {e}\n")
            rows.append({"status":"FAIL_TRAIN","backbone":spec.backbone,"selector":spec.selector,"seed":spec.seed,
                         "error":repr(e),"train_seconds":None,"eval_seconds":None,"device":_device_str()})
            counts["FAIL_TRAIN"] += 1
            continue

        print(f"[{i}/{len(grid)}] 🧪  EVAL   {tag}")
        te0 = time.perf_counter()
        try:
            metrics = evaluate_clean(model, backbone=spec.backbone, device=_device_str(), temperature=random_temperature)
            t_eval = time.perf_counter() - te0
            row = {
                "status": metrics.get("status","OK"),
                "backbone": spec.backbone,
                "selector": spec.selector,
                "seed": spec.seed,
                "device": _device_str(),
                "train_seconds": round(t_train, 3),
                "eval_seconds": round(t_eval, 3),
                "macro_f1_val": metrics.get("macro_f1_val"),
                "macro_f1_test": metrics.get("macro_f1_test"),
                "bce_val": metrics.get("bce_val"),
                "bce_test": metrics.get("bce_test"),
                "img_dim": metrics.get("img_dim"),
                "txt_dim": metrics.get("txt_dim"),
                "lbl_dim": metrics.get("lbl_dim"),
                "n_val": metrics.get("n_val"),
                "n_test": metrics.get("n_test"),
                "view": metrics.get("view"),
                "normalize": metrics.get("normalize"),
                "threshold": metrics.get("threshold"),
                "batch_size": metrics.get("batch_size"),
                "random_temperature": (random_temperature if spec.selector=="random" else None),
            }
            rows.append(row)
            if row["status"] == "OK":
                counts["OK"] += 1
                print(f"[{i}/{len(grid)}] ✅  OK     {tag}  | F1_test={row['macro_f1_test']:.4f} | BCE_test={row['bce_test']:.6f}\n")
            else:
                counts["FAIL_EVAL"] += 1
                print(f"[{i}/{len(grid)}] ⚠️  EVAL WARN {tag}  (status={row['status']})\n")
        except Exception as e:
            print(f"[{i}/{len(grid)}] ❌ EVAL FAILED {tag} :: {type(e).__name__}: {e}\n")
            rows.append({"status":"FAIL_EVAL","backbone":spec.backbone,"selector":spec.selector,"seed":spec.seed,
                         "error":repr(e),"train_seconds":round(t_train,3),"eval_seconds":None,"device":_device_str(),
                         "random_temperature": (random_temperature if spec.selector=="random" else None)})
            counts["FAIL_EVAL"] += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # write JSON
    with (artifacts_dir / "results.json").open("w") as f:
        json.dump(rows, f, indent=2)

    # write CSV
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with (artifacts_dir / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # write META
    meta = {
        "created_at_paris": _now_paris().isoformat(),
        "device": _device_str(),
        "torch_version": torch.__version__,
        "num_runs": len(rows),
        "summary": {"OK": counts["OK"], "FAIL_TRAIN": counts["FAIL_TRAIN"], "FAIL_EVAL": counts["FAIL_EVAL"], "total": len(rows)},
        "grid": [{"backbone": g.backbone, "selector": g.selector, "seed": g.seed} for g in grid],
        "paths": {"json": str(artifacts_dir / "results.json"), "csv": str(artifacts_dir / "results.csv")},
        "selector_space": ["shared", "random"],
        "notes": f"RandomSelection uses softmax(rand/temperature), temperature={random_temperature:g}. Lower is peakier.",
    }
    with (artifacts_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print("================================ SUMMARY ================================")
    print(f"OK={counts['OK']}  FAIL_TRAIN={counts['FAIL_TRAIN']}  FAIL_EVAL={counts['FAIL_EVAL']}  total={len(rows)}")
    print(f"Saved JSON: {artifacts_dir / 'results.json'}")
    print(f"Saved CSV : {artifacts_dir / 'results.csv'}")
    print(f"Saved META: {artifacts_dir / 'meta.json'}")
    print("=========================================================================\n")

    return {"artifacts_dir": str(artifacts_dir), "rows": rows, "summary": meta["summary"]}


# ---------- optional main (no CLI; just run defaults) ----------
def main():
    out = orchestrate_attgw(
        backbones=("clip", "blip2"),
        selectors=("shared", "random"),
        seeds=(42, 1337, 2025),
        results_root="./attgw_results",
        random_temperature=0.1,  # make it 'more extreme'
    )
    print("Artifacts directory:", out["artifacts_dir"])

if __name__ == "__main__":
    main()
