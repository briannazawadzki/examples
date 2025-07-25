import torch
import matplotlib
import matplotlib.pyplot as plt
import time
import numpy as np
from common_data import imager, model, dataset, vel, channel, tot_flux
from common_functions import train
from astropy.io import fits
from astropy.visualization import SqrtStretch
import pandas as pd

###################################################################################################
# Params to change
###################################################################################################

# initialize config
epochs  = 250000
ent_val = 0
tot_flux = tot_flux[0] #clean image in Jy, LkCa15
tsv_val = 1e-4
spa_val = 1e-5
tv_val  = 0

rml_filestring = "channel"+str(channel)
npz_string = rml_filestring + ".npz"
fitsname = rml_filestring + ".fits"

if ent_val==0:
    ent_str = "0"
else:
    ent_str = "{:.1e}".format(ent_val)
if tsv_val==0:
    tsv_str = "0"
else:
    tsv_str = "{:.1e}".format(tsv_val)
if spa_val==0:
    spa_str = "0"
else:
    spa_str = "{:.1e}".format(spa_val)

config = {'lr': 5.0,
        'entropy': ent_val, 'prior_intensity': 1.0, 'tot_flux':tot_flux,
        'lambda_sparsity': spa_val,
        'lambda_TV': tv_val,
        'lambda_TSV': tsv_val, 
        'epochs': epochs}

print(config)

ent_string = r'$\lambda_{\rm{ent}} = $' + ent_str
tsv_string = r'$\lambda_{\rm{TSV}} = $' + tsv_str
spa_string = r'$\lambda_{\rm{sparse}} = $' + spa_str

vel_string = r'$v=${:.2f}'.format(vel)

###################################################################################################

# get the dirty image so we can make the comparison img later
img, beam = imager.get_dirty_image(weighting="briggs", robust=0.5, unit="Jy/arcsec^2") # Jy/arcsec^2
img = 1000*img # mJy/arcsec^2

# query to see if we have a GPU
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

dset = dataset.to(device)
model = model.to(device)

print(device)
tic = time.perf_counter()
# initialize model to trained dirty image
model.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))

# create an optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma = 0.9, verbose=True)
writer = None

# run the training loop
loss_val, loss_track = train(model, dataset, optimizer, scheduler, config, device=device, writer=writer)
toc = time.perf_counter()

print("Elapsed time {:} s".format(toc - tic))

###################################################################################################
data_rml = np.squeeze(model.icube.sky_cube.detach().cpu().numpy()) # Jy/arcsec^2
data_rml = 1000*data_rml # mJy/arcsec^2

##########################################################################################

# Comparison plot with convergence info (convergence, dirty, RML)

#calculate moving average
n=1000
loss_running_avg = pd.Series(loss_track).rolling(window=n).mean().iloc[n-1:].values
print(loss_running_avg)

fig, ax = plt.subplots(ncols=3, figsize=(12, 4))

ax[0].plot(loss_track-(np.min(loss_track)))
ax[0].plot(loss_running_avg-(np.min(loss_track)), color='black')
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Loss")
ax[0].set_yscale("log")
ax[0].set_title("Convergence Check")

im = ax[1].imshow(
    img[0], # mJy/arcsec^2
    cmap='inferno',
    origin="lower",
    interpolation="none",
    extent=model.icube.coords.img_ext
)

im = ax[2].imshow(
    data_rml, #mJy/arcsec^2
    origin="lower",
    cmap='inferno',
    interpolation="none",
    extent=model.icube.coords.img_ext
)

ax[1].set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
ax[1].set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
ax[1].set_title("Dirty Image")

ax[2].set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
ax[2].set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
ax[2].set_title("Regularized Image")

plt.tight_layout()

fig.subplots_adjust(right=0.98)

pathstring = "converge_compare" + rml_filestring + ".png"
fig.savefig(pathstring, dpi=300)

##########################################################################################

np.savez(
    npz_string,
    img=np.squeeze(model.icube.sky_cube.detach().cpu().numpy()),
    ext=model.icube.coords.img_ext,
)

model.icube.to_FITS(fname=fitsname, overwrite=True) #by default calls the file cube.fits
