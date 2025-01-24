import matplotlib.pyplot as plt 
import numpy as np
import argparse
import load_data
import torch

from mpol import fourier, utils

def main():
    parser = argparse.ArgumentParser(
        description="Plot the true spectrum of the image vs. q."
    )

    parser.add_argument("outfile", help="Output file") 
    args = parser.parse_args()

    # get max baseline
    q_data = torch.hypot(load_data.vis_data.uu, load_data.vis_data.vv)
    q_max = torch.max(q_data)

    # use pre-defined variables in load_data
    load_data.packed_cube 

    # set coords 
    coords = load_data.coords

    flayer = fourier.FourierCube(coords)
    # pass through layer to set internal state
    flayer(load_data.packed_cube)
    
    amp = utils.torch2npy(flayer.ground_amp.flatten())
    qs = coords.ground_q_centers_2D.flatten()

    fig, ax = plt.subplots(nrows=1, figsize=(6.0,4.3))
    ax.axvline(q_max * 1e-6, lw=0.5, zorder=-1)
    ax.scatter(qs *1e-6, amp, s=0.4, rasterized=True, linewidths=0.0, c="k", alpha=0.2)
    ax.set_xlabel(r"$q$ [M$\lambda$]")
    ax.set_ylabel(r"$|V|$ [Jy]")
    ax.set_yscale("log")
    fig.subplots_adjust(left=0.18, right=0.82, bottom=0.12, top=0.92)
    ax.set_title("True spectrum")
    fig.savefig(args.outfile, dpi=300)

if __name__=="__main__":
    main()
