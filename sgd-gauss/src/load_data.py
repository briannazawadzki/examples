import numpy as np
import torch
import visread.process

from mpol import coordinates, fourier, utils
from pathlib import Path

from numpy import typing as npt
from typing import Any

from dataclasses import dataclass

# assumes you're running this from sgd/, not sgd/src/
_npz_path = Path("data/mock_data.npz")

# _npz_path = files(".").joinpath("mock_data.npz")
_nchan = 1
_cell_size = 0.005

archive = np.load(_npz_path)
img2D_butterfly = archive["img"]
npix, _ = img2D_butterfly.shape

sky_cube = torch.unsqueeze(torch.from_numpy(img2D_butterfly), dim=0)
packed_cube = utils.sky_cube_to_packed_cube(sky_cube)
# nchan = 1 for this example, so unsqueeze is simplest to add channel dimension
# now (1, npix, npix)

# stored as (nvis,)
# broadcast to (nchan, nvis)
uu_m, vv_m = archive["uu"], archive["vv"]
uu_lam, vv_lam = visread.process.broadcast_and_convert_baselines(
    uu_m, vv_m, np.array([230.0e9])
)

# stored as 1D, broadcast to (nchan, nvis)
w = archive["weight"]

# because we took only a fraction of the data, the original weight produces very 
# noisey visibilities.
# We will reduce sigma by a factor s
s = 10.0
# which increases weight by a factor s^2
w *= s**2
weight = np.broadcast_to(w, uu_lam.shape)

# set up image and fourier coordinates
coords = coordinates.GridCoords(cell_size=_cell_size, npix=npix)

# set random seed to make dataset repeatable
torch.manual_seed(42)

# convert to float32
uu_lam = np.float32(uu_lam)
vv_lam = np.float32(vv_lam)
weight = np.float32(weight)

# fake data routine expects tensors and returns a tensor
data, _ = fourier.generate_fake_data(
    packed_cube,
    coords,
    torch.as_tensor(uu_lam),
    torch.as_tensor(vv_lam),
    torch.from_numpy(weight.copy()),
)


# package everything up for easy use in other routines using dataclasses
@dataclass
class VisData:
    uu: torch.Tensor
    vv: torch.Tensor
    weight: torch.Tensor
    data: torch.Tensor

# store everything as 1D
vis_data = VisData(
    torch.squeeze(torch.from_numpy(uu_lam)),
    torch.squeeze(torch.from_numpy(vv_lam)),
    torch.squeeze(torch.from_numpy(weight.copy())),
    torch.squeeze(data),
)