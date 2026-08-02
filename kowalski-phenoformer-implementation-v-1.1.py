# @Author: Ben Kowalski
# Original published version from Garnot et al. as published in "Deep learning meets tree phenology modelling: PhenoFormer versus process-based models"
# Purpose: This is a recreation of the PhenoFormer model as published in the paper above. In this function I aim to retrain previously trained model, 
#          then provide my own attempt at training a model to predict the same phenophases. This enables future comparison of my implementation and 
#          the provided implementation that was rewritten in proven-phenoformer-implementation.py.


## IMPORTS
#import json
import os
import warnings
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, Subset

from configs.PROBLEM_CONFIG import target_list_parser
from dataset import ClimatePhenoDataset
from model.architecture import PhenoFormer
from train import LitModel

warnings.filterwarnings('ignore')

# GLOBAL VARIABLES

# create checkpoint directory to store data
checkpoint_dir = "pf_checkpoints/"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "transformer_weights.pt")

#load in data
dataset_folder = Path("data/PhenoFormer-data/learning-models-data")
target_list = ['Small_leaved_lime:leaf_unfolding', 'Horse_chestnut:leaf_unfolding', 'Large_leaved_lime:leaf_unfolding', 'Common_rowan:leaf_unfolding', 'European_white_birch:leaf_unfolding', 'European_beech:leaf_unfolding', 'Hazel:leaf_unfolding', 'European_larch:needle_emergence', 'Common_spruce:needle_emergence']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
retrain_flag = False
running_on_9070xt = True
batch_size = 16
if running_on_9070xt:
    torch.set_float32_matmul_precision('high') # utilize my specific GPU better, enables higher performance at cost of precision

def structured_temporal_split(dt, train_years_to=2002, val_years_to=2012):
    """Chronological split: train <= train_years_to, test > val_years_to, val between.

    Mirrors `--split_mode structured` in train.py (the `structured_temporal`
    entry in configs/RUN_CONFIGS.py). Deterministic -- no shuffle, no seed.
    """
    years = np.array(dt.years).astype(int)
    train_idxs = list(np.where(years <= train_years_to)[0])
    test_idxs = list(np.where(years > val_years_to)[0])
    val_idxs = sorted(set(range(len(years))) - set(train_idxs) - set(test_idxs))
    return train_idxs, val_idxs, test_idxs

# Helper functions
# initialise PhenoFormer model with required config
def _initialise_phenoformer(config):
    target_list = target_list_parser(config['target'])
    model = PhenoFormer(
        target_list=target_list,
        d_in=7,
        d_out=1,
        d_model=config['d_model'],
        nhead=config['nhead'],
        dim_feedforward=config['dim_feedforward'],
        n_layers=config['n_layers'],
        elevation=config['elevation'],
        latlon=config['latlon'],
    )
    model = LitModel(
        backbone=model,
        target_scaler=None,
        args=Namespace(**config)
    )
    return model, target_list

# move information to devices
def _move_batch_to_device(obj, device):
    """Recursively move all tensors in a (possibly nested) batch to specified device."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_batch_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_batch_to_device(v, device) for v in obj)
    return obj

def save_checkpoint(model, path=checkpoint_path):
    torch.save(model.state_dict(), path)
    print(f"Saved weights to {path}")

def load_checkpoint(model, path=checkpoint_path, device=device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"Loaded weights from {path}")
    return model


def evaluate_model(model, data, device, batch_size=64, num_workers=0):
    """Evaluate `model` across the full dataset and report accuracy in days as listed in our targets for the assignment.

    For every site-year we collect the observed phenophase (missing ones removed)
    and my model's prediction, then accumulate them per phenophase, then compute
    RMSE / MAE / R2 over that population.

    Returns a DataFrame indexed by phenophase with columns: n, rmse, mae, r2.
    """
    # Resolve the base dataset for metadata (a Subset stores it on `.dataset`).
    base_dt = data.dataset if isinstance(data, Subset) else data

    model = model.to(device)
    model.eval()
    loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    preds_by_target = {t: [] for t in base_dt.target_list}
    trues_by_target = {t: [] for t in base_dt.target_list}

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            preds = model.predict_unnormalised_dates(batch)  # predictions already in days
            targets = batch['target']
            for t in base_dt.target_list:
                true_norm = targets[t]
                valid = true_norm != base_dt.nan_value_target  # skip unobserved species
                if valid.sum() == 0:
                    continue
                mean, std = base_dt.target_scaler[t]
                true_days = true_norm[valid] * std + mean  # un-normalise only valid targets
                pred_days = preds[t][valid]
                trues_by_target[t].append(true_days.detach().cpu())
                preds_by_target[t].append(pred_days.detach().cpu())

    rows = {}
    for t in base_dt.target_list:
        if len(trues_by_target[t]) == 0:
            rows[t] = {'n': 0, 'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}
            continue
        true = torch.cat(trues_by_target[t]).numpy()
        pred = torch.cat(preds_by_target[t]).numpy()
        err = pred - true
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((true - true.mean()) ** 2))
        rows[t] = { 'n': len(true), #num of calculations for target
            'rmse': float(np.sqrt(np.mean(err ** 2))), # RMSE
            'mae': float(np.mean(np.abs(err))), # MAE
            'r2': float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan, # R2
        }

    return pd.DataFrame.from_dict(rows, orient='index', columns=['n', 'rmse', 'mae', 'r2'])




# Section 1 of code review - used functionality to ensure the data set was clean and organized
# create version of dataset with no jittering
dt = ClimatePhenoDataset(
    folder=dataset_folder,
    target_list=target_list,
    normalise_climate=True,
    normalise_dates=True,
    nan_value_climate=0,
    nan_value_target=-1000,
)
# sigma jittered data to reduce overfitting and improve generalization of the model
jitter_dt = ClimatePhenoDataset(
    folder=dataset_folder,
    target_list=target_list,
    normalise_climate=True,
    normalise_dates=True,
    nan_value_climate=0,
    nan_value_target=-1000,
    sigma_jitter=0.1
)

# seed all RNGs
np.random.seed(421999)
pl.seed_everything(seed=421999, workers=True)

# split data with structured split to ensure equal phenophase distribution
train_idxs, val_idxs, test_idxs = structured_temporal_split(dt)

train_loader = DataLoader(Subset(jitter_dt, train_idxs), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
val_loader   = DataLoader(Subset(dt, val_idxs),   batch_size=batch_size, shuffle=False,  drop_last=False,num_workers=0)
test_loader  = DataLoader(Subset(dt, test_idxs),  batch_size=batch_size, shuffle=False,  drop_last=False, num_workers=0)

# create model and prepare to train
args = Namespace(loss="L2", optim="adamw", learning_rate=1e-4, wd=0, nan_value_target=-1000)  # must match dataset's nan_value_target

# my version of the PhenoFormer model
kowalski_model = PhenoFormer(
    target_list=target_list,
    d_in=7,
    d_out=1,
    d_model=128,
    nhead=8,
    dim_feedforward=128,
    n_layers=2,
    elevation=True,
    latlon=False,
)


model = LitModel(backbone=kowalski_model, target_scaler=dt.target_scaler, args=args)
accelerator = "gpu" if torch.cuda.is_available() else "cpu"
# create logger for tracking training metrics
csv_logger = CSVLogger(save_dir="logs", name="kowalski_phenoformer", flush_logs_every_n_steps=5)
# create a more robust training mechanism
ckpt_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="kowalski_phenoformer-best-model-observed",
    monitor="val/loss",
    mode="min",
    save_top_k=1,
)

early_callback = EarlyStopping(
    monitor="val/loss", # NOTE: We locked the configure optimizer to val/rmse in train.py()
    patience=15,
    mode="min"
)

trainer = pl.Trainer(max_epochs=300,
                     min_epochs=50,
                    accelerator=accelerator,
                    devices=1 if torch.cuda.is_available() else None,
                    logger=csv_logger,
                    log_every_n_steps=5,
                    callbacks=[ckpt_callback, early_callback])  # Running using available device
if os.path.exists(checkpoint_path) and not retrain_flag:
    load_checkpoint(kowalski_model)
    # Move parameters if loading from checkpoint
    for meter_group in model.meters.values():
        for t in meter_group:
            meter_group[t] = meter_group[t].to(device)
    
else:
    trainer.fit(model, train_loader, val_loader)
    checkpoint = torch.load(ckpt_callback.best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    save_checkpoint(kowalski_model)

    # track the training curve
    metrics_path = Path(csv_logger.log_dir) / "metrics.csv"
    metrics_df = pd.read_csv(metrics_path)
    train_curve = (
        metrics_df[["epoch", "train/loss", "train/rmse", "val/loss"]]
        .groupby("epoch")
        .mean()
    )
    print(f"\nLogged training curve (from {metrics_path}):")
    print(train_curve.round(3))

trainer.test(model, test_loader) # output the Loss on the test set for various losses

test_dt = Subset(dt, test_idxs)
results = evaluate_model(model, test_dt, device, batch_size=64)
print(results.round(2))  # per-phenophase RMSE / MAE / R2 in days

# Macro-average across phenophases (each phenophase weighted equally, as in the paper),
# plus a sample-weighted average that weights each phenophase by its number of observations.
macro = results[['rmse', 'mae', 'r2']].mean()
weights = results['n']
micro_rmse = float(np.sqrt(np.average(results['rmse'] ** 2, weights=weights)))
print("\nMacro-average across phenophases (equal weight):")
print(macro.round(2))
print(f"\nSample-weighted RMSE across all observations: {micro_rmse:.2f} days")