# @Author: Ben Kowalski
# Original published version from Garnot et al. as published in "Deep learning meets tree phenology modelling: PhenoFormer versus process-based models"
# Purpose: This is a recreation of the PhenoFormer model as published in the paper above. In this function I aim to retrain previously trained model, 
#          then provide my own attempt at training a model to predict the same phenophases. This enables future comparison of my implementation and 
#          the provided implementation that was rewritten in proven-phenoformer-implementation.py.


## IMPORTS
#import json
import math
import os
import warnings
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset

from configs.PROBLEM_CONFIG import target_list_parser
from dataset import ClimatePhenoDataset, spatio_temporal_split
from model.architecture import PhenoFormer
from train import LitModel

warnings.filterwarnings('ignore')

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
    """Recursively move all tensors in a (possibly nested) batch to `device`."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_batch_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_batch_to_device(v, device) for v in obj)
    return obj

def provide_unnormalised_predicted_dates(predicted_dates, unnormalised_target_dates, target_list, target_dates, dt, model):
    with torch.no_grad():
        failed_list = []
        for target_name in target_list:
            if target_dates[target_name]==dt.nan_value_target:
                #unnormalised_target_dates[target_name] = np.nan
                failed_list.append(target_name)
            else:
                mean, std = dt.target_scaler[target_name]
                unnormalised_target = target_dates[target_name] * std + mean 
                unnormalised_target_dates[target_name] = int(unnormalised_target)
                predicted_dates[target_name] = model.predict_unnormalised_dates(data)[target_name]

    return predicted_dates, unnormalised_target_dates, failed_list


# Step 1: Load dataset
#load in data
dataset_folder = Path("data/PhenoFormer-data/learning-models-data") #adjusted from original path
target_list = ['Small_leaved_lime:leaf_unfolding', 'Horse_chestnut:leaf_unfolding', 'Large_leaved_lime:leaf_unfolding', 'Common_rowan:leaf_unfolding', 'European_white_birch:leaf_unfolding', 'European_beech:leaf_unfolding', 'Hazel:leaf_unfolding', 'European_larch:needle_emergence', 'Common_spruce:needle_emergence']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
retrain_flag = False
running_on_9070xt = True
if running_on_9070xt:
    torch.set_float32_matmul_precision('high') # utilize my specific GPU better, enables higher performance at cost of precision



# Section 1 of code review - used functionality to ensure the data set was clean and organized
dt = ClimatePhenoDataset(
    folder=dataset_folder,
    target_list=target_list,
    normalise_climate=True,
    normalise_dates=True,
    nan_value_climate=0,
    nan_value_target=-1000,
    sigma_jitter=0.05
)

# Split dataset into training and testing sets based on temporal split
train_idxs, val_idxs, test_idxs = spatio_temporal_split(dt, ratio=0.7)
# create model and prepare to train
args = Namespace(loss="L2", optim="adam", learning_rate=1e-3, wd=0,
                 nan_value_target=-1000)  # must match your dataset's nan_value_target

#section 2 of code review - used functionality to ensure the model would train in a reasonable time on my machine
# my version of the PhenoFormer model using lower numbers of internal dimensions to simplify procesing
kowalski_model = PhenoFormer(
    target_list=target_list,
    d_in=7,
    d_out=1,
    d_model=16,
    nhead=2,
    dim_feedforward=32,
    n_layers=1,
    elevation=True,
    latlon=False,
)
# create checkpoint directory to store data
checkpoint_dir = "pf_checkpoints/"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "transformer_weights.pt")

def save_checkpoint(model, path=checkpoint_path):
    torch.save(model.state_dict(), path)
    print(f"Saved weights to {path}")

def load_checkpoint(model, path=checkpoint_path, device=device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"Loaded weights from {path}")
    return model


train_loader = DataLoader(Subset(dt, train_idxs), batch_size=16, shuffle=True,  num_workers=0)
val_loader   = DataLoader(Subset(dt, val_idxs),   batch_size=16, shuffle=False, num_workers=0)
test_loader  = DataLoader(Subset(dt, test_idxs),  batch_size=16, shuffle=False, num_workers=0)

model = LitModel(backbone=kowalski_model, target_scaler=dt.target_scaler, args=args)
accelerator = "gpu" if torch.cuda.is_available() else "cpu"
trainer = pl.Trainer(max_epochs=35, accelerator=accelerator, log_every_n_steps=5)  # UPDATE: Running using available device
if os.path.exists(checkpoint_path) and not retrain_flag:
    load_checkpoint(kowalski_model)
    # Move parameters if loading from checkpoint
    for meter_group in model.meters.values():
        for t in meter_group:
            meter_group[t] = meter_group[t].to(device)
    
else:
    trainer.fit(model, train_loader, val_loader)
    save_checkpoint(kowalski_model)

trainer.test(model, test_loader) # output the Loss on the test set for various losses

# create comparitive prediction set to the PhenoFormer model
dataset_folder = Path("data/PhenoFormer-data/learning-models-data") #adjusted from original path
dt = ClimatePhenoDataset(
    folder=dataset_folder,
    target_list=target_list,
)
model = model.to(device)
model.eval()
data_loader = torch.utils.data.DataLoader(dt, batch_size=1, shuffle=True)

data = data_loader.__iter__().__next__()
data = _move_batch_to_device(data, device) # pass data to the same device as the model

target_dates = data['target']
predicted_dates,unnormalised_target_dates, failed_list = provide_unnormalised_predicted_dates({}, {}, dt.target_list, target_dates, dt, model)
i = 5
while failed_list:
    data = data_loader.__iter__().__next__()
    data = _move_batch_to_device(data, device)

    predicted_dates, unnormalised_target_dates, failed_list = provide_unnormalised_predicted_dates(predicted_dates, unnormalised_target_dates, failed_list, data['target'], dt, model)

print(predicted_dates)
print(unnormalised_target_dates)


## Show predicted vs True observed phenophase dates 
out = pd.DataFrame(index =dt.target_list, columns=['predicted', 'ground truth', 'RMSE'], dtype=float)
for idx in out.index:
    kowalski_model_prediction = float(predicted_dates[idx].detach().cpu().item())
    ground_truth = unnormalised_target_dates[idx]
    out.loc[idx] = [kowalski_model_prediction, ground_truth, math.sqrt((kowalski_model_prediction - ground_truth)**2)]
    
print(out.round(1)) #print generated results rounded to 1 decimal place
print(math.fsum(out['RMSE']) / len(out['RMSE']))