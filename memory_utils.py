# @Author: Ben Kowalski
# Purpose: Teardown / VRAM release helpers for the PhenoFormer training driver.
#
# Target stack (verified against .venv):
#   pytorch_lightning 2.6.5 | torch 2.9.1+rocm7.2.1 | torchmetrics 1.9.0
#
# NOTE ON ROCm: torch.cuda.* is the correct namespace on ROCm/HIP builds.
# torch.cuda.is_available() returns True for the 9070 XT. No hip-specific
# calls are needed.
#
# ---------------------------------------------------------------------------
# WHY THIS MODULE EXISTS
# ---------------------------------------------------------------------------
# torch.cuda.empty_cache() alone does almost nothing after a Lightning run.
# The caching allocator will only return a block to the driver once the last
# Python reference to the tensor is gone. After trainer.fit() there are three
# things still holding references:
#
#   1. trainer.optimizers[0].state  -> AdamW exp_avg + exp_avg_sq, ~2x params
#   2. Trainer <-> LightningModule  -> a genuine reference CYCLE, so plain
#                                      refcounting never collects it; only
#                                      gc.collect() breaks it
#   3. the loaded checkpoint dict   -> if map_location pointed at the GPU
#
# That is why every helper below frees references FIRST and calls
# empty_cache() LAST, and why free_cuda() runs gc.collect() before it.
#
# ---------------------------------------------------------------------------
# LIMITATION - READ THIS
# ---------------------------------------------------------------------------
# No function here can free an object the CALLER still has a name bound to.
# Passing `checkpoint` into release_all() does not release it; the caller must
# `del checkpoint` in its own scope. These helpers clear internal/nested
# references (optimizer state, metric buffers, loader workers) and then run the
# collector. Scoping your run inside a function (see managed_run) is what
# actually makes the top-level names go away.

from __future__ import annotations

import gc
from contextlib import contextmanager

import torch

__all__ = [
    "cuda_available",
    "cuda_snapshot",
    "free_cuda",
    "release_optimizer_state",
    "release_dataloaders",
    "release_meters",
    "release_module",
    "release_trainer",
    "release_all",
    "eval_mode",
    "managed_run",
]

_MB = 1024 ** 2


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------

def cuda_available() -> bool:
    """True on CUDA and on ROCm/HIP builds alike."""
    return torch.cuda.is_available()


def cuda_snapshot(tag: str = "", verbose: bool = True) -> dict:
    """Return (and optionally print) current accelerator memory usage.

    Take one of these before training, one after fit, and one after cleanup.
    Without a baseline you cannot tell whether a fix did anything, and
    'allocated' vs 'reserved' is the distinction that matters:

      allocated  - bytes held by live tensors. This is the number that proves
                   a reference was actually dropped.
      reserved   - bytes the caching allocator holds from the driver. Only
                   empty_cache() moves this, and it does not imply a leak.

    Returns {} when no accelerator is present, so it is safe to call on CPU.
    """
    if not cuda_available():
        if verbose:
            print(f"[vram:{tag:<20}] no accelerator")
        return {}

    stats = {
        "tag": tag,
        "allocated_mb": torch.cuda.memory_allocated() / _MB,
        "reserved_mb": torch.cuda.memory_reserved() / _MB,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / _MB,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / _MB,
    }
    if verbose:
        print(
            f"[vram:{tag:<20}] "
            f"alloc {stats['allocated_mb']:9.1f} MiB | "
            f"reserved {stats['reserved_mb']:9.1f} MiB | "
            f"peak alloc {stats['peak_allocated_mb']:9.1f} MiB"
        )
    return stats


def free_cuda(reset_peak: bool = False) -> None:
    """Collect Python garbage, then return cached blocks to the driver.

    Order is not optional. gc.collect() must run first because the
    Trainer <-> LightningModule cycle is unreachable by refcounting; if you
    call empty_cache() first, those tensors are still live and nothing is
    returned.

    reset_peak=True zeroes the peak counters, which is what you want between
    a training phase and an evaluation phase if you are measuring each
    separately.
    """
    gc.collect()
    if not cuda_available():
        return
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass  # not implemented on every ROCm build
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Targeted release helpers
# ---------------------------------------------------------------------------

def release_optimizer_state(trainer, verbose: bool = True) -> int:
    """Drop optimizer moment buffers. Biggest single post-fit win.

    AdamW keeps exp_avg and exp_avg_sq per parameter, so its state is roughly
    2x the model itself, and it lives wherever the params live. Lightning
    holds the optimizer on trainer.optimizers for the entire life of the
    Trainer object -- long after fit() returns and through test().

    Returns bytes freed, so you can report it.
    """
    freed = 0
    for opt in list(getattr(trainer, "optimizers", None) or []):
        try:
            for state in opt.state.values():
                for value in state.values():
                    if torch.is_tensor(value):
                        freed += value.numel() * value.element_size()
            opt.state.clear()
        except Exception as exc:
            if verbose:
                print(f"[cleanup] optimizer state not cleared: {exc!r}")

    # Trainer.optimizers has a setter that forwards to the strategy.
    # Trainer.lr_scheduler_configs is READ-ONLY (no setter in PL 2.6.5), so it
    # must be cleared on the strategy directly.
    try:
        trainer.optimizers = []
    except Exception:
        pass
    try:
        trainer.strategy.lr_scheduler_configs = []
    except Exception:
        pass

    if verbose and freed:
        print(f"[cleanup] optimizer state released: {freed / _MB:.1f} MiB")
    return freed


def release_dataloaders(*loaders, verbose: bool = True) -> int:
    """Shut down any live worker processes and drop the iterator.

    A no-op at num_workers=0 (your current setting). It matters the moment you
    raise num_workers: a DataLoader that was interrupted mid-epoch keeps its
    _MultiProcessingDataLoaderIter alive, and with it the worker processes and
    their pinned-memory staging buffers.
    """
    closed = 0
    for loader in loaders:
        if loader is None:
            continue
        iterator = getattr(loader, "_iterator", None)
        if iterator is None:
            continue
        try:
            iterator._shutdown_workers()
            closed += 1
        except Exception:
            pass
        try:
            loader._iterator = None
        except Exception:
            pass
    if verbose and closed:
        print(f"[cleanup] shut down workers for {closed} dataloader(s)")
    return closed


def release_meters(lit_model, verbose: bool = True) -> int:
    """Reset every torchmetric on the module and move it off the accelerator.

    LitModel.meters is a plain dict-of-dicts, so PyTorch does not know these
    are submodules: they are not moved by .to(), not captured by state_dict(),
    and not freed when the module is. 27 metric objects (9 targets x 3 metrics)
    otherwise sit on the GPU until the interpreter exits.

    Written to work unchanged if you later convert meters to nn.ModuleDict --
    both expose .values().
    """
    meters = getattr(lit_model, "meters", None)
    if meters is None:
        return 0

    count = 0
    try:
        groups = meters.values()
    except AttributeError:
        return 0

    for group in groups:
        try:
            metrics = group.values()
        except AttributeError:
            metrics = group
        for metric in metrics:
            try:
                metric.reset()   # clears accumulated state + the _computed cache
                metric.to("cpu")
                count += 1
            except Exception:
                pass

    if verbose and count:
        print(f"[cleanup] reset and offloaded {count} metric(s)")
    return count


def release_module(module, verbose: bool = True) -> None:
    """Move a module to CPU, drop its gradients, and break the Trainer cycle.

    Setting .grad = None rather than calling zero_grad() actually frees the
    gradient tensors instead of filling them with zeros -- gradients are the
    same size as the parameters.
    """
    if module is None:
        return
    try:
        for param in module.parameters(recurse=True):
            param.grad = None
    except Exception:
        pass
    try:
        module.cpu()
    except Exception:
        pass
    # LightningModule._trainer is one half of the reference cycle.
    if hasattr(module, "_trainer"):
        try:
            module._trainer = None
        except Exception:
            pass
    if verbose:
        print(f"[cleanup] released module {type(module).__name__}")


def release_trainer(trainer, verbose: bool = True) -> None:
    """Tear down a Lightning Trainer: optimizers, strategy, loader references.

    Order matters. release_optimizer_state() runs FIRST because it actually
    frees the moment buffers; strategy.teardown() only calls
    _optimizers_to_device(optimizers, "cpu"), which moves them off VRAM but
    keeps them alive in host RAM. Clearing first makes the move a no-op.

    strategy.teardown() is otherwise the supported path -- it also does
    lightning_module.cpu(), precision_plugin.teardown() and
    accelerator.teardown(). It asserts accelerator is not None, so it raises on
    a Trainer that never ran; that is caught and reported, not fatal.

    Everything after it is best-effort and guarded, because these are
    private-ish attributes whose names have moved between Lightning versions.
    """
    if trainer is None:
        return

    release_optimizer_state(trainer, verbose=verbose)

    try:
        trainer.strategy.teardown()
    except Exception as exc:
        if verbose:
            print(f"[cleanup] strategy.teardown() skipped: {exc!r}")

    for attr in (
        "train_dataloader",
        "val_dataloaders",
        "test_dataloaders",
        "predict_dataloaders",
    ):
        try:
            setattr(trainer, attr, None)
        except Exception:
            pass

    # ModelCheckpoint holds best_model_score as a live tensor; the path string
    # is what you actually want to keep, so null the tensor only.
    for callback in getattr(trainer, "callbacks", None) or []:
        for attr in ("best_model_score", "kth_value", "current_score"):
            if hasattr(callback, attr):
                try:
                    setattr(callback, attr, None)
                except Exception:
                    pass

    if verbose:
        print("[cleanup] released trainer")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def release_all(
    model=None,
    trainer=None,
    dataloaders=(),
    reset_peak: bool = False,
    verbose: bool = True,
) -> dict:
    """Run the full teardown in dependency order and report what it recovered.

    Order matters: trainer first (it owns the optimizer and holds the model),
    then the module's metrics, then the module, then loaders, then collect.

    Reminder from the module header: this cannot free names the caller still
    holds. After calling it, `del` your own references -- particularly the
    checkpoint dict:

        release_all(model=model, trainer=trainer, dataloaders=[train_loader])
        del checkpoint
        free_cuda()

    Returns {"before": ..., "after": ..., "recovered_mb": float}.
    """
    before = cuda_snapshot("before cleanup", verbose)

    if trainer is not None:
        release_trainer(trainer, verbose=verbose)
    if model is not None:
        release_meters(model, verbose=verbose)
        release_module(model, verbose=verbose)
    if dataloaders:
        release_dataloaders(*dataloaders, verbose=verbose)

    free_cuda(reset_peak=reset_peak)
    after = cuda_snapshot("after cleanup", verbose)

    recovered = 0.0
    if before and after:
        recovered = before["allocated_mb"] - after["allocated_mb"]
        if verbose:
            print(f"[cleanup] recovered {recovered:.1f} MiB of allocated VRAM")

    return {"before": before, "after": after, "recovered_mb": recovered}


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------

@contextmanager
def eval_mode(model, inference: bool = False):
    """Set eval mode + disable autograd, and RESTORE the previous train flag.

    evaluate_model() currently calls model.eval() and never undoes it, so any
    code after it silently runs with dropout and batchnorm disabled. The
    finally block is the point of this wrapper.

    inference=True uses torch.inference_mode(), which is faster than no_grad
    but tags the outputs as inference tensors. Those raise if they are later
    used in an autograd-recording op. evaluate_model() calls torch.cat() on
    its collected tensors AFTER the loop exits, so leave this False unless you
    move that concatenation inside the block.
    """
    was_training = model.training
    model.eval()
    guard = torch.inference_mode() if inference else torch.no_grad()
    try:
        with guard:
            yield model
    finally:
        model.train(was_training)


@contextmanager
def managed_run(tag: str = "run", reset_peak: bool = True):
    """Wrap a whole training phase so cleanup happens even on OOM or Ctrl-C.

    Usage:

        with managed_run("fit"):
            trainer.fit(model, train_loader, val_loader)

    Without the finally block, an OOM part-way through epoch 40 leaves the
    optimizer state, activations and loader workers pinned, and the retry you
    immediately attempt OOMs earlier than the first one did.

    This handles the accelerator cache and the collector. It does NOT release
    objects created inside the block -- see release_all for those.
    """
    if cuda_available() and reset_peak:
        torch.cuda.reset_peak_memory_stats()
    cuda_snapshot(f"{tag}: enter")
    try:
        yield
    finally:
        cuda_snapshot(f"{tag}: exit")
        free_cuda()
        cuda_snapshot(f"{tag}: freed")


if __name__ == "__main__":
    # Smoke test: exercises every path on CPU with a throwaway model.
    import torch.nn as nn

    net = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    net(torch.randn(4, 64)).sum().backward()
    opt.step()

    class _FakeTrainer:
        def __init__(self, optimizer):
            self.optimizers = [optimizer]
            self.lr_scheduler_configs = []
            self.callbacks = []

    with managed_run("smoke"):
        with eval_mode(net) as m:
            assert not m.training
        assert net.training, "eval_mode failed to restore the train flag"

    stats = release_all(model=net, trainer=_FakeTrainer(opt), dataloaders=[None])
    assert opt.state == {}, "optimizer state was not cleared"
    print("memory_utils smoke test OK ->", stats["recovered_mb"], "MiB")
