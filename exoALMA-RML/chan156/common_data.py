import re
import os
def get_chan_from_dir(s):
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None

# Must have the channel number at the end of the directory
# e.g. channel100, chan100, cv100, etc.
curr_dir = os.getcwd()
channel = get_chan_from_dir(curr_dir)

#############################################################################
# Edit the following before running
#############################################################################
target = "LkCa15"
line = "12CO"

npz_visibilities = "../LkCa15_12CO_100ms_LSRK_contsub.npz"
crossval_method = "RandCell" 
k=10 # number of k-folds for cross-validation

#############################################################################
import numpy as np
import torch
from mpol import (
    coordinates,
    crossval,
    gridding,
    precomposed
)
from mpol.constants import *
from mpol.utils import loglinspace
import itertools

print("Working on " + target + ", " + line)
print("Channel " + str(channel))
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
    print("Warning: running on cpu")

# Load the data in read-only, memory efficient mode.
d = np.load(npz_visibilities, mmap_mode="r")

if line=="12CO":
    f0 = 3.4579598990e11 # 12CO rest frequency, Hz (345.79598990 GHz)
elif line=="13CO":
    f0 = 3.3058796530e11 # 13CO rest frequency, Hz (330.58796530 GHz)
elif line=="CS":
    f0 = 3.4288285030e11 #   CS rest frequency, Hz (342.88285030 GHz)
else:
    raise Exception("Please provide a line used in exoALMA, e.g. 12CO (case-sensitive)")

def get_velocity(rest, measured):
    return(c.to('km/s')*(rest-measured)/(rest)) # returns velocity in km/s

measured = d["freqs"][channel]
vel = get_velocity(f0, measured) # km/s

# subselect channel
data = np.ndarray.flatten(d["data"][channel]) # Jy
weight = d["weight"] # (nvis) 1/Jy^2
uu = np.ndarray.flatten(d["uu"]) # nvis [m]
vv = np.ndarray.flatten(d["vv"]) # nviz [m]

# set up pixel size and npix
npix=1024
cell_size=0.0125 # half that of the clean fits cube

coords = coordinates.GridCoords(cell_size=cell_size, npix=npix)
gridder = gridding.DataAverager(
    coords=coords,
    uu=uu,
    vv=vv,
    weight=weight,
    data_re=data.real,
    data_im=data.imag,
)
imager = gridding.DirtyImager(
    coords=coords,
    uu=uu,
    vv=vv,
    weight=weight,
    data_re=data.real,
    data_im=data.imag,
)

dataset = gridder.to_pytorch_dataset(check_visibility_scatter=False)

# create the model
model = precomposed.SimpleNet(coords=coords, nchan=dataset.nchan)

########################################################################
# Cross-Validation Setup
########################################################################

# Optional, can set a random seed for reproduceability
randseed = 13

if crossval_method=="RandCell":
    cv = crossval.RandomCellSplitGridded(dataset, k, seed=randseed)
elif crossval_method=="Dartboard":
    q_max=gridder.coords.q_max
    # 48 chunks total, default 12
    q_edges = loglinspace(0, q_max, N_log=30, M_linear=19)
    # 32 chunks (really 64 since it's from 0 to pi), default 8
    phi_edges = np.linspace(0, np.pi, num=32 + 1)
    dartboard = datasets.Dartboard(coords=coords, q_edges=q_edges, phi_edges=phi_edges)
    cv = crossval.DartboardSplitGridded(dataset, k, dartboard=dartboard, seed=randseed) # create cross validator using this "dartboard"
else:
    raise Exception("Please enter a valid CV partitioning scheme")

########################################################################
k_fold_datasets = [(train, test) for (train, test) in cv]
########################################################################

learning_rates = [0.5]
lambda_entropy = [0,8e-8,2e-7,8e-7,2e-6,8e-6,2e-5]
entropy_prior_intensity = [1.0]
tot_flux = [11.0] # total flux of the clean image in Jy
lambda_sparsity = [5e-5,1e-4,5e-4,1e-3,1e-2]
lambda_TV = [0]
lambda_TSV = [5e-6,1e-5,5e-5]
epochs = [20000] # number of iterations for cross-validation. Can be adjusted based on chosen learning rate, etc.
config_values = [learning_rates, lambda_entropy, entropy_prior_intensity, tot_flux, lambda_sparsity, lambda_TV, lambda_TSV, epochs]

all_configs = list(itertools.product(*config_values))

config_dicts = []
for i in range(len(all_configs)):
    temp_dict = {"lr": all_configs[i][0],
            "entropy": all_configs[i][1], "prior_intensity": all_configs[i][2], "tot_flux": all_configs[i][3],
            "lambda_sparsity": all_configs[i][4], 
            "lambda_TV": all_configs[i][5], 
            "lambda_TSV": all_configs[i][6], 
            "epochs": all_configs[i][7]
            }
    config_dicts.append(temp_dict)