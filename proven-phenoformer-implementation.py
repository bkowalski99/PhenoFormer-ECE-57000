# @Autho: Ben Kowalski
# Original published version from Garnot et al. as published in "Deep learning meets tree phenology modelling: PhenoFormer versus process-based models"
# Purpose: This is a recreation of the PhenoFormer model as published in the paper above. In this function I aim to load the pre-trained model, 
#          then utilize the model to predict the same phenophases. This enables future comparison of my implementation and the provided.


## IMPORTS
import os 
import json
import torch
import warnings

from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from configs.PROBLEM_CONFIG import target_list_parser
from dataset import ClimatePhenoDataset
from model.architecture import PhenoFormer
from train import LitModel

warnings.filterwarnings('ignore')


### CODE FROM THE ORIGINAL PHENOFORMER IMPLEMENTATION

## DEFINE HELPER FUNCTIONS
# loads configuration and summary from the files located in path
def _get_config_and_summary(path):
    summary_files = [f for f in os.listdir(path) if f.endswith('.json')]
    with open(path / summary_files[0]) as file:
        config = json.loads(file.read())
    return config

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

def _load_weights(model, path):
    weight_files = [f for f in os.listdir(path) if f.endswith('.ckpt')]
    weight_path = path / weight_files[0]

    checkpoint = torch.load(weight_path, map_location=torch.device('cpu')) #adjusted to load on CPU for compatibility
    model.load_state_dict(checkpoint['state_dict'])
    model.target_scaler = checkpoint['hyper_parameters']['target_scaler']
    
    return model 

def load_model(path):
    config = _get_config_and_summary(path)
    phenoformer, target_list = _initialise_phenoformer(config)
    phenoformer = _load_weights(phenoformer, path)
    return phenoformer , target_list
    

# Pre-loading weights, enabling comparison set

# Specify the path to one of the configurations (i.e. one of the subfolders of the "pre-trained-weights" folder)
path_to_model_folder = Path("./pre-trained-weights/MultiTask-phenoformer_default-LU+NE-uniformly_rdm-default/")

# Load model and set in evaluation mode 
trained_phenoformer, target_list = load_model(path_to_model_folder)
trained_phenoformer.eval()

print("Pre-trained PhenoFormer ready, list of predicted phases:")
print(target_list)


#load in data
dataset_folder = Path("data/PhenoFormer-data/learning-models-data") #adjusted from original path
dt = ClimatePhenoDataset(
    folder=dataset_folder,
    target_list=target_list,
)
data_loader = torch.utils.data.DataLoader(dt, batch_size=1, shuffle=True)

print(f"Dataset ready with {len(dt)} site-years")


## Show visual examples of data
# Grab sample
data = data_loader.__iter__().__next__()

grouping = [1,0,2,0,0,0,3]

# Convert tensors to numpy for plotting
climate = data['climate'][0].numpy()
doys = data['doys'][0].numpy()
elevation = data['elevation'][0].item()
lat, lon = data['latlon'][0].numpy()
year = data['year'][0].item()

# Create the plot
fig, ax = plt.subplots(4, 1, figsize=(12, 6), sharex=True)

# Plot each climate variable over the days of the year
for i in range(climate.shape[1]):
    climate_time_series = climate[:, i]
    variable_name = dt.var_names[i]
    mean= dt.norm_values[variable_name]['mean']
    std = dt.norm_values[variable_name]['std']
    
    climate_time_series = climate_time_series * std + mean # Un-normalise for visualisation
    
    ax[grouping[i]].plot(doys, climate_time_series, label=variable_name[:-1])

# Add static information as text on the plot
info_text = (
    f"Year: {int(year)}\n"
    f"Elevation: {elevation} m\n"
    f"Lat, Lon: ({lat:.4f}, {lon:.4f})"
)
ax[0].text(
    0.02, 1.5, info_text, 
    transform=ax[0].transAxes, fontsize=10,
    verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.1)
)

# Add labels and legend
ax[0].set_title('Climate Variables Over Days of the Year')
ax[0].set_ylabel('Value')
ax[0].legend(loc='upper right', ncol=2, fontsize='small')
ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)
ax[3].grid(True)

ax[0].set_ylabel('Temperature')
ax[1].set_ylabel('Photoperiod')
ax[2].set_ylabel('Precipitation')
ax[3].set_ylabel('Pressure')
ax[3].set_xlabel('Day of Year (DOY)')

# Show plot
plt.show()

## Get the predictions of PhenoFormer on that sample 
with torch.no_grad():
    predicted_dates = trained_phenoformer.predict_unnormalised_dates(data)

## Get the target phenophase dates (from the Swiss Phenology Network)
# and un-normalise them. 
target_dates = data['target']
unnormalised_target_dates = {}
for target_name in dt.target_list:
    if target_dates[target_name]==dt.nan_value_target:
        unnormalised_target_dates[target_name] = np.NaN
    else:
        mean, std = dt.target_scaler[target_name]
        unnormalised_target = target_dates[target_name] * std + mean 
        unnormalised_target_dates[target_name] = int(unnormalised_target)

## Show predicted vs True observed phenophase dates 
out = pd.DataFrame(index =dt.target_list, columns=['predicted', 'target'], dtype=float)
for idx in out.index:
    out.loc[idx] = [float(predicted_dates[idx].numpy()), unnormalised_target_dates[idx]]
    
print(out.round(1))

