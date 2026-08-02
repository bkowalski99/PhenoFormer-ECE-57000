# Memory Review — `kowalski-phenoformer-implementation-v-1.1.py`

**Scope:** driver script only. Items requiring edits to `train.py` or `model/architecture.py` are listed in the appendix but not fixed here.

**Stack (verified against `.venv`):** `pytorch_lightning 2.6.5` · `torch 2.9.1+rocm7.2.1` · `torchmetrics 1.9.0`
Note that `requirements.txt` still pins `torch==1.11.0` / `pytorch_lightning==1.5.10`. It is stale by two major Lightning versions and should not be trusted when reasoning about API behaviour.

**Companion module:** `memory_utils.py`

---

## Verdict first

There is **no unbounded per-step leak** in this script. Nothing accumulates tensors in a Python list during training, the torchmetric states are bounded scalar accumulators, and logged values are detached (by accident — see item A2 in the appendix).

What you have is **retention**: several large objects stay resident on the accelerator long after they are needed, and because everything lives at module scope, none of them ever go out of scope. In `rocm-smi` that is indistinguishable from a leak. Peak VRAM ends up being the *sum* of the training phase and the evaluation phase rather than the max of the two.

The distinction matters for how you fix it. A leak needs the accumulation site found. Retention needs scoping and explicit teardown — which is what the rest of this document is.

---

## Item table

| # | Item | Line(s) | Mechanism | Fix | Cleanup call |
|---|------|---------|-----------|-----|--------------|
| D1 | Checkpoint dict loaded onto GPU, never freed | 218 | `map_location=device` pulls `state_dict` **and** `optimizer_states` into VRAM; the global is never dropped | `map_location="cpu"`, then `del` | `free_cuda()` |
| D2 | Entire script at module scope | all | No name ever goes out of scope, so refcounts never hit zero | Wrap body in `main()` | — (scoping *is* the fix) |
| D3 | AdamW moment buffers retained after `fit` | 203, 217 | `trainer.optimizers[0].state` holds `exp_avg` + `exp_avg_sq` ≈ 2× params, on device, for the Trainer's whole life | Clear optimizer state post-fit | `release_optimizer_state(trainer)` |
| D4 | Trainer ↔ LightningModule reference cycle | 217 | Refcounting alone never collects a cycle; `empty_cache()` before `gc.collect()` returns nothing | Break the cycle, then collect | `release_trainer(trainer)` → `free_cuda()` |
| D5 | Five DataLoaders alive at once, eval at 4× batch size | 182–184, 107, 244 | train/val/test loaders stay live while `evaluate_model` opens a sixth at `batch_size=64` vs. training's 16 | Scope loaders; release before eval | `release_dataloaders(train_loader, val_loader)` |
| D6 | Metric objects stranded on GPU | 212–214 | `LitModel.meters` is a plain dict → not submodules → not moved, not freed with the module. 9 targets × 3 metrics = 27 objects | Reset + offload explicitly | `release_meters(model)` |
| D7 | `model.eval()` set and never restored | 106 | `evaluate_model` leaves the module in eval mode permanently | Use a restoring context manager | `with eval_mode(model):` |
| D8 | No `try/finally` around `fit` | 217 | An OOM or Ctrl-C at epoch 200 leaves optimizer state, activations and workers pinned; the retry OOMs *earlier* than the first attempt | Wrap the phase | `with managed_run("fit"):` |
| D9 | Full `dt` + three `Subset`s held through eval | 152, 182–184, 243 | Host RAM, not VRAM — the whole climate array stays live for the entire run | Scope inside `main()` | — |
| D10 | `predict_unnormalised_dates` mutates the backbone's dict in place | 117 | Correctness landmine, not memory. Safe today because the dict is rebuilt each forward; breaks silently if the backbone ever caches | Copy before mutating | — |
| D11 | No memory instrumentation | — | You cannot tell whether any of the above fixes did anything | Snapshot at phase boundaries | `cuda_snapshot(tag)` |

---

## Detail and suggested fixes

### D1 — Checkpoint dict on the GPU *(highest payoff, smallest diff)*

```python
# current, line 218
checkpoint = torch.load(ckpt_callback.best_model_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["state_dict"])
```

A Lightning 2.x checkpoint is a dict containing `state_dict`, `optimizer_states`, `lr_schedulers`, `loops`, and `hyper_parameters`. Because `save_hyperparameters()` in `train.py` was called without `ignore=`, `hyper_parameters` also holds a **pickled duplicate of the backbone** (appendix A1). `map_location=device` puts all of that in VRAM, and `checkpoint` is a module global that survives through `trainer.test()` and `evaluate_model()`.

```python
# fixed
checkpoint = torch.load(ckpt_callback.best_model_path, map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["state_dict"])
del checkpoint
free_cuda()
```

Load to CPU and let `load_state_dict` do the transfer — it copies into the already-resident parameters, so there is no reason for the source to be on device. `del` is required: `free_cuda()` cannot reach a name you still hold.

### D2 — Module scope

Everything from line 152 down runs at import time and never goes out of scope. Wrapping in `main()` makes the locals collectable at return, and is also the prerequisite for `num_workers > 0` (worker processes re-import the module; unguarded module-level training code will recurse).

### D3 / D4 — Optimizer state and the Trainer cycle

These two are why `torch.cuda.empty_cache()` on its own appears to do nothing. The optimizer moments are reachable from `trainer`, and `trainer` is reachable from `model._trainer` while `model` is reachable from `trainer.strategy._lightning_module`. That cycle is invisible to refcounting; only the cyclic collector breaks it. Hence the ordering baked into `free_cuda()`: `gc.collect()` **then** `empty_cache()`.

```python
from memory_utils import release_all, free_cuda

# after fit + test, before evaluate_model
release_all(model=model, trainer=trainer, dataloaders=[train_loader, val_loader])
del checkpoint, trainer, train_loader, val_loader
free_cuda(reset_peak=True)
model.to(device)          # release_module moved it to CPU; bring back for eval
```

### D5 — Concurrent loaders

`evaluate_model` builds its own loader at `batch_size=64`, four times the training batch size, while the training loaders are still live. Reuse `test_loader` or release the training loaders first. At `num_workers=0` the loaders themselves are cheap; the real cost is the 4× activation footprint of the larger eval batch landing on top of a not-yet-released training footprint.

### D6 — Stranded metrics

Lines 212–214 are a hand-rolled device loop:

```python
for meter_group in model.meters.values():
    for t in meter_group:
        meter_group[t] = meter_group[t].to(device)
```

That loop exists only because `meters` is a plain dict rather than an `nn.ModuleDict`, so PyTorch does not know these are submodules. The proper fix is in `train.py` (appendix A3). Driver-side, `release_meters(model)` resets and offloads all 27, which is the containment measure until you make that change. `release_meters` is written to work either way.

### D7 — Restore the train flag

```python
# in evaluate_model, replace: model.eval() ... with torch.no_grad():
from memory_utils import eval_mode

with eval_mode(model):
    for batch in loader:
        ...
```

### D8 / D11 — Exception safety and instrumentation

```python
from memory_utils import managed_run, cuda_snapshot

cuda_snapshot("baseline")
with managed_run("fit"):
    trainer.fit(model, train_loader, val_loader)
```

---

## Suggested `main()` structure

```python
from memory_utils import (
    cuda_snapshot, free_cuda, release_all, eval_mode, managed_run,
)

def main():
    cuda_snapshot("baseline")

    dt = ClimatePhenoDataset(...)
    np.random.seed(421999)
    train_idxs, val_idxs, test_idxs = spatio_temporal_split(dt, ratio=0.7)

    kowalski_model = PhenoFormer(...)
    model = LitModel(backbone=kowalski_model, target_scaler=dt.target_scaler, args=args)

    train_loader = DataLoader(Subset(dt, train_idxs), batch_size=16, shuffle=True)
    val_loader   = DataLoader(Subset(dt, val_idxs),   batch_size=16, shuffle=False)
    test_loader  = DataLoader(Subset(dt, test_idxs),  batch_size=16, shuffle=False)

    trainer = pl.Trainer(...)

    if os.path.exists(checkpoint_path) and not retrain_flag:
        load_checkpoint(kowalski_model)
        model.to(device)
    else:
        with managed_run("fit"):                                     # D8
            trainer.fit(model, train_loader, val_loader)

        checkpoint = torch.load(                                     # D1
            ckpt_callback.best_model_path, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["state_dict"])
        save_checkpoint(kowalski_model)
        del checkpoint
        free_cuda()

        metrics_df = pd.read_csv(Path(csv_logger.log_dir) / "metrics.csv")
        print(metrics_df[["epoch", "train/loss", "train/rmse", "val/rmse"]]
              .groupby("epoch").mean().round(3))

    with managed_run("test"):
        trainer.test(model, test_loader)

    # D3/D4/D5/D6 -- hand back everything the training phase was holding
    release_all(model=model, trainer=trainer,
                dataloaders=[train_loader, val_loader, test_loader])
    del trainer, train_loader, val_loader
    free_cuda(reset_peak=True)

    model.to(device)                                                 # eval needs it back
    with managed_run("evaluate"):
        results = evaluate_model(model, Subset(dt, test_idxs), device, batch_size=64)

    print(results.round(2))
    macro = results[["rmse", "mae", "r2"]].mean()
    micro_rmse = float(np.sqrt(np.average(results["rmse"] ** 2, weights=results["n"])))
    print(macro.round(2))
    print(f"\nSample-weighted RMSE: {micro_rmse:.2f} days")

    release_all(model=model)
    return results


if __name__ == "__main__":
    main()
```

`if __name__ == "__main__"` is not optional once `num_workers > 0`.

---

## Verifying the fix actually worked

`cuda_snapshot` reports both numbers on purpose:

- **`allocated`** — bytes held by live tensors. This is the one that proves a reference was dropped. If `allocated` after cleanup is not meaningfully lower than before, something still holds a name.
- **`reserved`** — bytes the caching allocator holds from the driver. Only `empty_cache()` moves it, and a high value is *not* evidence of a leak. `rocm-smi` shows you this number, which is why it overstates the problem.

Expected pattern with the fixes in:

```
[vram:baseline            ] alloc     0.0 MiB
[vram:fit: exit           ] alloc   XXX.X MiB   <- params + grads + AdamW moments
[vram:before cleanup      ] alloc   XXX.X MiB
[vram:after cleanup       ] alloc     ~0 MiB    <- D3/D4 landed
[vram:evaluate: enter     ] alloc     YY.Y MiB  <- params only
```

If `after cleanup` does not fall near zero, the remaining holder is almost always a name in the caller's scope that was never `del`'d.

---

## Appendix — out of scope for this pass

These are real, but require touching files outside the driver.

| # | Item | File | Note |
|---|------|------|------|
| A1 | `save_hyperparameters()` called with no `ignore` | `train.py:33` | Pickles the whole backbone **and** `target_scaler` into `self.hparams` and into every checkpoint file. Fix: `self.save_hyperparameters(ignore=["backbone", "target_scaler"])`. Lightning normally warns about this — `warnings.filterwarnings('ignore')` at driver line 28 is suppressing it. |
| A2 | `self.log(name, value, kwargs)` | `train.py:185` | Third positional parameter is `prog_bar`, not `**kwargs`. Your `on_step` / `on_epoch` are silently discarded and `prog_bar` receives a truthy dict. Fix: `self.log(name, value, **kwargs)`. Memory-relevant because `enable_graph` defaults to `False` — that default is the only reason logged training metrics aren't dragging the autograd graph into epoch-end aggregation. Fix the call, but do **not** pass `enable_graph=True`. |
| A3 | `self.meters` is a plain dict | `train.py:47` | Root cause of D6 and of the device-placement hacks. Fix: nested `nn.ModuleDict`. Changes `state_dict` keys, so existing checkpoints need `strict=False` once or a retrain. |
| A4 | `PositionalEncoder.denom` is a raw attribute with a one-shot device latch | `architecture.py` | Not a registered buffer, so not in `state_dict` and not moved by `.to()`. `self.updated_location` means it moves exactly once, ever — move the model to a second device and you either crash or strand a live allocation. Fix: `register_buffer("denom", ...)` and drop the latch. This is upstream Garnot et al. code, so the change diverges your fork. |
| A5 | `logs/` and `lightning_logs/` accumulate a `version_N` per run | — | Disk, not memory. `run_output.txt` is already 36 MB. |
