# End-to-end setup steps (orchestrator-only)

1) Create repo skeleton — completed
   - git init, add MIT LICENSE, .gitignore, README.

2) Scaffold Poetry project — completed
   - Python 3.11; shimmer pinned to git@github.com:ruflab/shimmer.git@main.
   - Deps pinned to working conda env (torch 2.8.0, lightning 2.5.0.post0, etc.).
   - Script entry: orchestrate-attgw = imdb_gw.training.final_orchestrator:main.

3) Layout source tree — completed
   - src/imdb_gw/{data,domains,training,overrides} with __init__.py.

4) Keep only orchestrator path — completed
   - Orchestrator stack: final_orchestrator.py, function_train_attgw.py, function_eval_attgw.py
   - Data/label prep: dataset_1x_aug70.py, make_mm_imdb1_clip_blip2_embeddings.py, make_mm_imdb2_clip_blip2_embeddings.py, augment_labels.py
   - Domains: domains_obb.py
   - Overrides: shimmer_patches.py

5) Monkeypatch shimmer — completed
   - shimmer_patches.py patches GlobalWorkspaceFusion, contrastive/broadcast losses, selectors; applied on import.

6) Imports rewritten — completed
   - All remaining scripts import via imdb_gw.* and load shimmer_patches.

7) Strip W&B — completed
   - No wandb imports; CSV/ModelCheckpoint only.

8) CLI — completed
   - Poetry script orchestrate-attgw points to final_orchestrator:main.

9) Docs — partially done
   - README + SETUP_PLAN present. TODO: optional example.env for data paths.

10) Housekeeping — completed
    - .gitignore present in repo.

11) Lock deps — completed
    - poetry install succeeded; poetry.lock added.

12) Smoke-check — TODO (requires data)
    - Run orchestrate-attgw with real/small data to confirm end-to-end.

13) Commit — TODO after locking deps/smoke-check.

14) Usage reminder
```bash
poetry install
# ensure data files exist where dataset_1x_aug70.py expects them
poetry run orchestrate-attgw
```
