import numpy as np
import torch
import visread.process

from mpol import coordinates, fourier, utils
from pathlib import Path

# assumes you're running this from sgd/, not sgd/src/
_npz_path = Path("data/mock_data.npz")

# _npz_path = files(".").joinpath("mock_data.npz")
_nchan = 1
_cell_size = 0.005

archive = np.load(_npz_path)
img2D_butterfly = np.float64(archive["img"])
npix, _ = img2D_butterfly.shape

sky_cube = torch.unsqueeze(torch.from_numpy(img2D_butterfly), dim=0)
packed_cube = utils.sky_cube_to_packed_cube(sky_cube)
# nchan = 1 for this example, so unsqueeze is simplest to add channel dimension
# now (1, npix, npix)

# stored as (nvis,)
# broadcast to (nchan, nvis)
uu_m, vv_m = np.float64(archive["uu"]), np.float64(archive["vv"])
uu_lam, vv_lam = visread.process.broadcast_and_convert_baselines(
    uu_m, vv_m, np.array([230.0e9])
)

# stored as 1D, broadcast to (nchan, nvis)
weight = np.broadcast_to(np.float64(archive["weight"]), uu_lam.shape)

# set up image and fourier coordinates
coords = coordinates.GridCoords(cell_size=_cell_size, npix=npix)

# fake data routine expects tensors
data, _ = fourier.generate_fake_data(
    packed_cube,
    coords,
    torch.as_tensor(uu_lam),
    torch.as_tensor(vv_lam),
    torch.from_numpy(weight.copy()),
)

# package everything up as a dataclass for easy use in other routines


# @pytest.fixture(scope="session")
# def mock_dataset_np(baselines_2D_np, weight_2D_t, mock_data_t):
#     uu, vv = baselines_2D_np
#     weight = utils.torch2npy(weight_2D_t)
#     data = utils.torch2npy(mock_data_t)
#     data_re = np.real(data)
#     data_im = np.imag(data)

#     return (uu, vv, weight, data_re, data_im)
