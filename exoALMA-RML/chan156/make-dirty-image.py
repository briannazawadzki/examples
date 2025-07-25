import numpy as np 
import matplotlib.pyplot as plt
from common_data import imager, gridder

img, beam = imager.get_dirty_image(weighting="briggs", robust=0.5, unit="Jy/beam")

kw = {"origin": "lower", "extent": gridder.coords.img_ext}
fig, ax = plt.subplots(ncols=1)
snp = ax.imshow(img[0], cmap='inferno', **kw)
ax.set_title("image")
ax.set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
ax.set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
plt.colorbar(snp)
fig.savefig("dirty_image.png", dpi=300)

kw = {"origin": "lower", "extent": gridder.coords.img_ext}
fig, ax = plt.subplots(ncols=1)
snp = ax.imshow(beam[0], cmap='inferno', **kw)
ax.set_title("beam")
ax.set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
ax.set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
plt.colorbar(snp)
r=1
ax.set_xlim(r, -r)
ax.set_ylim(-r, r)
fig.savefig("dirty_beam.png", dpi=300)

