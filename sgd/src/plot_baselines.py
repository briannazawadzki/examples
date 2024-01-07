import matplotlib.pyplot as plt 
import numpy as np
import argparse
import load_data

from mpol import utils

def main():
    parser = argparse.ArgumentParser(
        description="Plot the visibilities."
    )

    parser.add_argument("outfile", help="Output file") 
    args = parser.parse_args()
    
    vis_data = load_data.vis_data
    
    uu = utils.torch2npy(vis_data.uu)
    vv = utils.torch2npy(vis_data.vv)

    # augment to include complex conjugates
    uu = np.concatenate([uu, -uu])
    vv = np.concatenate([vv, -vv])

    fig, ax = plt.subplots(nrows=1, figsize=(4.5,4.3))
    ax.scatter(uu *1e-6, vv*1e-6, s=1.5, rasterized=True, linewidths=0.0, c="k")
    ax.set_xlabel(r"$u$ [M$\lambda$]")
    ax.set_ylabel(r"$v$ [M$\lambda$]")
    ax.axis('equal')
    fig.subplots_adjust(left=0.18, right=0.82, bottom=0.12, top=0.92)
    ax.set_title("Baseline distribution")
    fig.savefig(args.outfile, dpi=120)

if __name__=="__main__":
    main()