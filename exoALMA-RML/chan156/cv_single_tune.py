import torch
import time
import numpy as np
import argparse
from common_data import coords, model, dataset, k, k_fold_datasets, config_dicts
import matplotlib.pyplot as plt

from mpol import (
    losses,
    fourier,
    precomposed
)

# query to see if we have a GPU
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
writer = None
print(device)

dset = dataset.to(device)
model = model.to(device)

parser = argparse.ArgumentParser(description="index of current dictionary")
parser.add_argument("dict_index", type=int, help="dictionary index containing config vals")
args = parser.parse_args()

config = config_dicts[args.dict_index]
print(config)
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

def train(
    model, dataset, optimizer, config, device, writer=None, report=False, logevery=50
): 
    """
    Args:
        model: neural net model
        dataset: to use to train against
        optimizer: tied to model parameters and used to take a step
        config: dictionary including epochs and hyperparameters.
        device: "cpu" or "cuda"
        writer: tensorboard writer object
    """
    model.train()

    loss_tracker = []
    for iteration in range(config["epochs"]):

        model.zero_grad()
        vis = model.forward()
        sky_cube = model.icube.sky_cube

        loss = (
            losses.nll_gridded(vis, dataset)
            + config["entropy"] * losses.entropy(sky_cube, config["prior_intensity"], config['tot_flux'])
            + config["lambda_sparsity"] * losses.sparsity(sky_cube)
            + config["lambda_TV"] * losses.TV_image(sky_cube)
            + config["lambda_TSV"] * losses.TSV(sky_cube)
        )

        loss_tracker.append(loss.item())

        loss.backward()
        optimizer.step()

    if report:
        tune.report(loss=loss.item())

    return loss.item(), loss_tracker

flayer = fourier.FourierCube(coords=coords)
flayer = flayer.to(device)
flayer.forward(torch.zeros(dset.nchan, coords.npix, coords.npix))

def test(model, dset):
    model.train(False)
    # evaluate test score
    vis = model.forward()
    loss = losses.nll_gridded(vis, dset)
    return loss.item()

def cross_validate(config, check_convergence=False):
    """
    config is a dictionary that should contain ``lr``, ``lambda_sparsity``, ``lambda_TV``, ``epochs``
    """
    test_scores = []


    fig = plt.figure(figsize=(5,25))
    gs = fig.add_gridspec(k, hspace=0.05)
    ax = gs.subplots(sharex=True, sharey=True)
    fig.suptitle('Convergence Check')
    ax[k-1].set_xlabel("Iteration")

    for k_fold, (train_dset, test_dset) in enumerate(k_fold_datasets):
        print("Working on k-fold k=" + str(k_fold))
        
        train_dset = train_dset.to(device)
        test_dset = test_dset.to(device)

        # create a new model and optimizer for this k_fold
        rml = precomposed.SimpleNet(coords=coords, nchan=train_dset.nchan)
        rml = rml.to(device)
        optimizer = torch.optim.Adam(rml.parameters(), lr=config["lr"])

        # train for a while
        loss_val, loss_track = train(rml.to(device), train_dset.to(device), optimizer, config, device=device)
        # evaluate the test metric
        test_scores.append(test(rml, test_dset))

        if np.min(loss_track)<0:
            ax[k_fold].plot(loss_track-np.min(loss_track), color='blue')
            ax[k_fold].set_yscale('log')
        else:
            ax[k_fold].plot(loss_track, color='green')
            ax[k_fold].set_yscale('log')
            ax[k_fold].set_ylim(np.min(loss_track),np.max(loss_track))
        ax[k_fold].set_ylabel("Loss")

    # aggregate all test scores and sum to evaluate crossval metric
    test_score = np.sum(np.array(test_scores))
    if check_convergence:
        pathstring = "all_convergence_check_config" + "{:d}".format(args.dict_index) + ".png"
        fig.savefig(pathstring, dpi=200)

    return test_score, test_scores

tic = time.perf_counter()
cv_score, ind_cv_scores = cross_validate(config, check_convergence=True)
toc = time.perf_counter()
print(ind_cv_scores)
print("Cross validation score:", cv_score)
print("Elapsed time {:} s".format(toc - tic))

cv_string = "Cross validation score: " + str(cv_score)

# various cross-validation outputs

with open('CV_summary_randcell_all.txt', 'a') as f:
        f.write('\n')
        config_string = "Config index " + str(args.dict_index) + ": " + str(config)
        f.write(config_string)
        f.write('\n')
        f.write(str(ind_cv_scores))
        f.write('\n')
        f.write(cv_string)
        f.write('\n-------------------------------------------------------------------')

with open('heatmap_ent_randcell_all.txt', 'a') as f:
        config_string = "[" + str(args.dict_index) + ", " + str(config["entropy"]) + "], "
        f.write(config_string)

with open('heatmap_tsv_randcell_all.txt', 'a') as f:
        config_string = "[" + str(args.dict_index) + ", " + str(config["lambda_TSV"]) + "], "
        f.write(config_string)

with open('heatmap_spa_randcell_all.txt', 'a') as f:
        config_string = "[" + str(args.dict_index) + ", " + str(config["lambda_sparsity"]) + "], "
        f.write(config_string)

with open('heatmap_cv_scores_randcell_all.txt', 'a') as f:
        config_string = "[" + str(args.dict_index) + ", " + str(cv_score) + "], "
        f.write(config_string)




