# Assignment 2 (Preliminaries) — Working Plan

**Track:** Tiny Reproductions — distill & verify PhenoFormer's core claim with a scaled-down
model + dataset.
**Deliverable (due tonight):** an **8-slide Quarto RevealJS markdown** deck. NOT a finished
system. The rubric explicitly says code "does not need to be perfect or fully runnable." You are
graded on: (1) Project Context, (2) Code Evidence, (3) Result & Analysis — equally weighted.

> This file is a plan/checklist only. It intentionally does NOT contain your slide text or a
> finished implementation — those are yours to write.

---

## 0. Proposal alignment (verified against `kowalski_proposal_ECE_57000.qmd`)

Your submitted proposal commits you to **"recreating, training, and evaluating the model"** —
not just running pre-trained weights. This means **Path B (training) is the correct fit** and
Path A (inference) is only a sanity check / backup. Locked-in numbers to carry onto your slides:

| Anchor | RMSE (days) | Role |
|---|---|---|
| Paper PhenoFormer (Spring) | **8.5** | reproduction target |
| Null model (Spring) | **13.6** | minimum-viable bar to beat |
| Your tiers (vs. paper's 8.5) | Excellent ≤9.5 · Good ≤10.5 · Okay ≤12.5 · Poor >12.5 | grading context (Slide 7/8) |

Scope you committed to: **Spring** phenology (`leaf_unfolding` / `needle_emergence`), single/subset
of species, simpler transformer if compute-limited. Keep everything below focused on Spring.

> Reality check: a tiny 1-epoch preliminary run will NOT hit 8.5 days — that is expected and fine.
> The preliminary number just proves the training loop runs; Slide 8 explains the gap + next steps.

---

## 0.1 Reframe the goal for tonight

The 8 slides only need:
- 2 context slides (problem + method) — mostly reuse your proposal.
- 2 code snippets (≤20 lines each) + 2 explanation slides.
- 1 preliminary result (a *single number* is explicitly allowed) + 1 analysis/next-steps slide.

So the minimum viable path is: **get ONE number that proves the pipeline runs**, then write the
deck around the code you already touched. Everything below optimizes for that.

---

## 1. Two candidate paths to a preliminary result (pick ONE)

### Path B — Train a *tiny* model end-to-end
This is the path your proposal commits to ("recreating, training, and evaluating"). Call
[train.py](../train.py) **directly, once** (one target, one fold, few epochs) — do NOT run
`cross_val_train.py` or the `run-phenoformer-*` scripts tonight (those loop 40 folds × species ×
splits = far too much).

Minimal single-run invocation (single species, spring, tiny model, few epochs, CPU):
```powershell
python train.py `
  --data_folder data/PhenoFormer-data/learning-models-data `
  --target BEE:LU `
  --split_mode structured --train_years_to 2002 --val_years_to 2012 `
  --d_model 16 --nhead 2 --dim_feedforward 32 --n_layers 1 `
  --batch_size 16 --learning_rate 0.001 --max_epochs 3 `
  --xp_name kowalski_tiny_smoketest --save_dir training_logs
```
- `--target BEE:LU` = `European_beech:leaf_unfolding` (uses `species_short`/`phases_short` codes;
  see `target_list_parser` in [configs/PROBLEM_CONFIG.py](../configs/PROBLEM_CONFIG.py)).
- `structured` split needs no `--fold`; a `.json` split (e.g. `splits/uniformly-rdm-split.json`)
  would require `--fold 1`.
- The run prints a test RMSE and writes a result summary under `save_dir`. That test RMSE (days)
  is your Slide-7 number.
- Report train/val/test RMSE; contextualize vs. Null (13.6) and paper (8.5).

### Path A — Inference from pre-trained weights (BACKUP only, not the thesis)
If training hits a blocker tonight, fall back to the [demo.ipynb](../demo.ipynb) inference flow
(repoint `dataset_folder` to `data/PhenoFormer-data/learning-models-data`, load a variant from
[pre-trained-weights](../pre-trained-weights), compute predicted-vs-observed RMSE). Use this only
as a stopgap Slide-7 number and be explicit on the slide that it's pre-trained inference, with
from-scratch training as the immediate next step.

**Recommendation:** Attempt Path B first (it IS your project). Keep Path A ready as a fallback so
you are guaranteed a number either way.

---

## 2. What "scaled-down" means for your Tiny Reproduction

Define the knobs you will turn (state these on the methodology slide). You do NOT have to run all
of them tonight — just declare the reduction strategy:
- **Fewer tasks:** single species, single phenophase (e.g. `European_beech:leaf_unfolding`)
  instead of the 9-species multi-task setup.
- **Smaller model:** `d_model` 64→16, `nhead` 8→2, `dim_feedforward` 128→32, `n_layers`=1.
- **Coarser input:** `monthly=True` in `ClimatePhenoDataset` → 12 timesteps instead of ~365
  daily steps (big compute reduction, already supported by the dataset class).
- **Fewer epochs / subset of site-years** for a fast smoke-test.
- **One split** (`uniformly_rdm`) instead of all four.

The reproduction *claim* you are verifying: "a small attention model predicts phenophase dates
from climate time series better than a trivial baseline (e.g. predict-the-mean-date)." That
mean-date baseline gives you a second easy number to contextualize your RMSE.

---

## 2.1 Your required "own modification" (per Piazza clarification)

Professor's directive for Tiny Reproductions: reusing the authors' open-source code is fine, but
you MUST apply **your own modification** — *"simplification, using another dataset, or trying edge
cases"* — and "put yourself in the shoes of the reviewer." This is what turns reuse into a
reproduction *contribution* and satisfies the rubric's "meaningful coding effort" line. State the
chosen modification explicitly on Slide 2. Pick ONE spine (✅ = tonight-feasible / ⏳ = later):

**Simplification**
- ✅ Monthly inputs (`monthly=True`) → 12 vs ~365 timesteps; report the accuracy cost.
- ✅ Temperature-only (subset the 7 climate vars) → how much signal lives in temperature alone.
- ✅ Tiny model (`d_model` 64→16, `nhead` 8→2) — already the §2 baseline.
- ⏳ Swap the transformer for a simpler encoder (mean-pool+MLP or one GRU) → tests whether
  attention is actually needed at tiny scale. Strong reviewer story; more code.

**Edge cases / stress tests (high reviewer value, low code)**
- ✅ Climate-shift generalization: `structured` split (train ≤2002, test >2012) → directly probes
  the paper's central claim about generalizing under climate shift.
- ⏳ Cross-species transfer; positional-encoding / climate-variable ablation; `sigma_jitter`
  noise-robustness sweep.

**Another dataset** ⏳ — hardest tonight (only the Swiss set is local); a pragmatic proxy is
holding out a region/elevation band as an out-of-distribution test.

**Recommended spine:** combine a cheap simplification with one edge case, e.g. *"a temperature-only,
monthly, tiny PhenoFormer, stress-tested under a temporal climate-shift split."* Both are
tonight-feasible, need little new code, and the edge case engages the paper's thesis — exactly the
reviewer framing the professor asked for. The scientific contribution is YOUR call; this is a
recommendation, not a prescription.

---

## 3. Files & functions to reuse (do not rewrite these)

| Need | File | What to use |
|---|---|---|
| Load & normalize data | [dataset.py](../dataset.py) | `ClimatePhenoDataset(folder, target_list, monthly=...)`; note `target_scaler` for un-normalizing dates |
| Model | [model/architecture.py](../model/architecture.py) | `PhenoFormer(...)`; the encoder/decoder core is `shared_linear_encoder` → `transformer` → `linear_decoder` in `forward()` |
| Transformer layer | [model/transformer_pytorch.py](../model/transformer_pytorch.py) | `TransformerEncoderLayer` (custom, supports `return_attention`) |
| Train/metrics wrapper | [train.py](../train.py) | `LitModel`, `predict_unnormalised_dates()`, RMSE/MAE/R² meters |
| Target parsing / names | [configs/PROBLEM_CONFIG.py](../configs/PROBLEM_CONFIG.py) | `target_list_parser`, species/phase tables, `spring_phenophases` |
| Configs | [configs/RUN_CONFIGS.py](../configs/RUN_CONFIGS.py) | `model_configs`, `training_configs['debug']`, `datasplit_configs` |
| End-to-end inference example | [demo.ipynb](../demo.ipynb) | `load_model`, predicted-vs-target `DataFrame` (Path A template) |

Your existing scratch file [kowalski-phenoformer-implementation.py](../kowalski-phenoformer-implementation.py)
currently only has imports — build Path A on top of it (or in a fresh notebook).

---

## 4. Choosing your two code snippets (≤20 lines each)

Pick snippets that show *your* engagement, not library imports (rubric penalizes trivial imports).
Good pairing:

- **Snippet 1 — Data loading / processing.** The `ClimatePhenoDataset` instantiation with your
  tiny settings (single target, `monthly=True`) plus how a sample is turned into
  `climate` / `doys` / `target` tensors. Explanation (Slide 4): why normalization + the adaptive
  spring/autumn time range matter, and how site-years are filtered to those with complete data.
- **Snippet 2 — Encoder→decoder core.** The ~6 key lines of `PhenoFormer.forward()`:
  `shared_linear_encoder` (per-timestep linear projection), `+ positional_encoding(doys)`,
  prepend `learnt_tokens`, `transformer(...)`, then `linear_decoder` on the task-token embeddings.
  Explanation (Slide 6): this is the reproduction's heart — learnt tokens act as per-phenophase
  "query" slots read out by the transformer; the linear decoder maps each token to a date.

This matches your stated plan (load dataset → parse/update → review linear encoder & decoder).

---

## 5. Tonight's ordered checklist (Path B primary)

1. [ ] Confirm the environment: `python model/architecture.py` (`__main__` smoke test builds
       PhenoFormer + dummy forward pass). If it prints predictions, torch/imports are fine.
2. [ ] Verify pinned versions are installed (see §7): `pytorch_lightning==1.5.10`, `torch==1.11.0`,
       `wandb==0.12.11`. A newer Lightning WILL break `train.py`.
3. [ ] Run the single tiny training invocation from §1 (Path B). Watch for the printed test RMSE.
4. [ ] Record train/val/test **RMSE in days** → this is your Slide-7 number.
5. [ ] Quote the **Null (mean-date) baseline = 13.6 days** and paper **= 8.5 days** for context,
       and place your result on the Excellent/Good/Okay/Poor tier ladder.
6. [ ] Save the loss/RMSE curve or a small results table → Slide 7 artifact.
7. [ ] Extract your 2 code snippets (trim each to ≤20 lines) → Slides 3 & 5.
8. [ ] Draft the 8 Quarto slides (see structure below); keep prose tight.
9. [ ] If Path B blocks: switch to Path A (pre-trained inference) for a fallback number and note
       the blocker + debug plan on Slide 8 (the rubric explicitly rewards this).

---

## 6. Slide-by-slide skeleton (fill with your own words)

1. **Problem & Goal** — species-level tree phenology prediction from climate series; goal =
   verify PhenoFormer's core claim at tiny scale (reuse proposal).
2. **Methodology** — attention model (learnt tokens + transformer + linear decoder); state the
   scaling-down knobs from §2.
3. **Code Snippet 1** — dataset loading (≤20 lines).
4. **Explain Snippet 1** — normalization, adaptive time range, site-year filtering.
5. **Code Snippet 2** — encoder→decoder core of `forward()` (≤20 lines).
6. **Explain Snippet 2** — learnt tokens as per-phenophase queries; linear readout to a date.
7. **Preliminary Result** — test **RMSE (days)** from your tiny few-epoch training run, plotted
   against Null (13.6) and paper (8.5); state which tier it lands in.
8. **Analysis & Next Steps** — what the RMSE means (gap to 8.5 and to the 13.6 bar), then next
   steps: more epochs / early stopping, full daily (vs. monthly) inputs, larger `d_model`,
   add more species, run all folds/splits.

---

## 7. Likely blockers & pre-emptive fixes (Path B focus)

- **PyTorch Lightning version (highest-risk):** `train.py` uses the pre-2.0 API
  (`pl.Trainer(gpus=...)` and `pl.Trainer.add_argparse_args`). These were REMOVED in Lightning
  2.x. You MUST run with the pinned `pytorch_lightning==1.5.10` + `torch==1.11.0`. If a newer
  Lightning is in your env, create the phenoformer env from [requirements.txt](../requirements.txt)
  (and `pip install pip==24.0` first, per the repo readme).
- **`wandb` dependency:** `train.py` uses `WandbLogger(..., offline=True)`. Offline avoids login,
  but `wandb==0.12.11` must be installed. If it causes trouble, the quickest workaround is to
  read the RMSE from the printed `trainer.test(...)` output rather than the wandb summary.
- **Windows `num_workers=8`:** the DataLoaders hard-code `num_workers=8`, which can stall or
  error on Windows. If startup hangs, that's the cause — a tiny run may still work, otherwise
  note it as the blocker (this is a legitimate Slide-8 roadblock).
- **`--target` format:** must be short codes `SPECIES:PHASE` (e.g. `BEE:LU`), not the long name.
- **CPU only:** `gpus=False` (GPU_ENABLED=False) already targets CPU; keep the model tiny and
  `--max_epochs` small (2–5) so a run finishes tonight.
- **No result?** The rubric allows Slide 7 to show the key error message + a systematic debug
  plan on Slide 8. Keep a log of what you tried so this remains a strong option.

---

## 8. Definition of done for tonight
- One reproducible **training** number: test RMSE (days) from a tiny few-epoch run, placed on
  the tier ladder vs. Null (13.6) and paper (8.5).
- Two ≤20-line snippets you can explain (data loading + encoder/decoder core).
- 8 Quarto slides drafted. Fallback secured: Path A inference number if training blocks.
