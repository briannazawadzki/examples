import numpy as np
import loaddata
import torch
import dirty_image
import argparse
import matplotlib.colors as mco
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, Sampler
from mpol import constants,coordinates, gridding, fourier, images, losses, utils, plot
import visread
import visread.visualization

from torch.utils.tensorboard import SummaryWriter

class Net(torch.nn.Module):
    r"""
    ObsID dict has keys of ObsIDs indexed from 0, with values lists of the DDIDs in that ObsID.
    """
    def __init__(
        self,
        coords=None,
        nchan=1,
        FWHM=0.05,
        obsid_dict=None,
        freeze_amps=False,
        freeze_weights=False,
        fixed_amp_index=0,
        freeze_positions=False,
        fixed_position_index=0
    ):
        super().__init__()

        self.coords = coords
        self.nchan = nchan
        self.fixed_amp_index = fixed_amp_index
        self.fixed_position_index = fixed_position_index

        # assumes ddids indexed from 0
        # mapping from ddid to obsid
        ddid2obsid = []
        for obsid, ddids in obsid_dict.items():
            for ddid in ddids:
                ddid2obsid.append(obsid)
        
        # use this for amplitude and astrometric offsets 
        self.register_buffer("ddid2obsid", torch.tensor(ddid2obsid))

        # one for each obsid, except for the fixed one
        self.log10_amp_factors = torch.nn.Parameter(torch.zeros(len(obsid_dict) - 1), requires_grad=(not freeze_amps))
        self.register_buffer("zero_val", torch.tensor([0.0]))

        # one for each ddid
        self.log10_sigma_factors = torch.nn.Parameter(torch.zeros(len(ddid2obsid)), requires_grad=(not freeze_weights))

        # positional offsets
        self.delta_ra = torch.nn.Parameter(torch.zeros(len(obsid_dict) - 1), requires_grad=(not freeze_positions))
        self.delta_dec = torch.nn.Parameter(torch.zeros(len(obsid_dict) - 1), requires_grad=(not freeze_positions))

        self.bcube = images.BaseCube(coords=self.coords, nchan=self.nchan)
        self.conv_layer = images.GaussConvFourier(coords=self.coords, FWHM_maj=FWHM, FWHM_min=FWHM)
        self.icube = images.ImageCube(coords=self.coords, nchan=self.nchan)
        self.nufft = fourier.NuFFT(coords=self.coords, nchan=self.nchan)

    def adjust_weights(self, ddid, weight):
        r"""
        Adjust the weights according to the rescale factor on the parameter.
        """

        log10_sigma_factors = self.log10_sigma_factors[ddid]
        
        return weight / (10**log10_sigma_factors)**2

    def adjust_amplitudes(self, ddid, vis, weight):
        r"""
        Adjust the model visibilities and weights according to the amplitude rescale factor.
        """

        # get the obsID that corresponds to all ddid
        obsid = self.ddid2obsid[ddid]

        # we have the problem that self.log10_amp_factors has (len(obsid) - 1) keys, 
        # so the index does not directly correspond to the factor,
        # and we need to index len(obs) unique keys

        # so our solution is to take the original array and insert 0 at the location of 
        # the fixed_amp_index
        # Unfortunately, PyTorch does not have an .insert method like Python lists, so workaround is

        # split self.log10_amp_factors at the index that should have the extra key
        split = torch.hsplit(self.log10_amp_factors, [self.fixed_amp_index])
        # reconcatenate with the key in the middle
        augmented_log10_amp_factors = torch.cat((split[0], self.zero_val, split[1]))

        # get the amp factors that correspond to each visibility
        amp_factors = 10**augmented_log10_amp_factors[obsid]
        
        vis_scaled = vis * amp_factors
        weight_scaled = weight / amp_factors**2

        return vis_scaled, weight_scaled
    
    def adjust_positions(self, ddid, uu, vv, vis):
        r"""
        Adjust the model visibilities according to the position offset factor.
        """
        # get the obsID that corresponds to all ddid
        obsid = self.ddid2obsid[ddid]

        # we have the problem that self.delta_ra and self.delta_delta have (len(obsid) - 1) keys, 
        # so the index does not directly correspond to the factor,
        # and we need to index len(obs) unique keys

        # so our solution is to take the original array and insert 0 at the location of 
        # the fixed_amp_index
        # Unfortunately, PyTorch does not have an .insert method like Python lists, so workaround is

        # split self.delta_ra at the index that should have the extra key
        split = torch.hsplit(self.delta_ra, [self.fixed_position_index])
        # reconcatenate with the key in the middle
        augmented_delta_ra = torch.cat((split[0], self.zero_val, split[1]))
        # get the factors that correspond to each visibility, depending on obsid
        delta_ra = augmented_delta_ra[obsid]

        split = torch.hsplit(self.delta_dec, [self.fixed_position_index])
        # reconcatenate with the key in the middle
        augmented_delta_dec = torch.cat((split[0], self.zero_val, split[1]))
        # get the factors that correspond to each visibility, depending on obsid
        delta_dec = augmented_delta_dec[obsid]

        # apply phase shift
        # convert from arcsec to radians
        delta_ra *= constants.arcsec
        delta_dec *= constants.arcsec

        shift_factor = torch.exp(-2.0j * np.pi * (delta_ra * uu + delta_dec * vv))
        vis_shift = vis * shift_factor

        return vis_shift
    

    def forward(self, uu, vv):
        r"""
        Predict model visibilities at baseline locations.

        Parameters
        ----------
        uu, vv : torch.Tensor
            spatial frequencies. Units of :math:`\lambda`.

        Returns
        -------
        torch.Tensor
            1D complex torch tensor of model visibilities.
        """
        # Feed-forward network passes base representation through "layers"
        # to create model visibilities
        x = self.bcube()
        x = self.conv_layer(x)
        x = self.icube(x)
        vis = self.nufft(x, uu, vv)
        return vis


class DdidBatchSampler(Sampler):
    """
    Custom Sampler to gather mini-batches having the same ddid (usually corresponds to a unique spw for a unique obs id).
    """
    def __init__(self, ddids, batch_size, shuffle=True):
        self.ddids = ddids
        self.ids = np.arange(len(ddids))
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.unique_ddids = np.unique(self.ddids)

        # calculate how many batches we will have for each ddid sub-group
        self.batches_per_ddid = {}
        for ddid in self.unique_ddids:
            # find all indices with ddid match
            ids_ddid = self.ids[(self.ddids == ddid)]
            # calculate the length
            nbatches = (len(ids_ddid) + self.batch_size - 1) // self.batch_size
            self.batches_per_ddid[ddid] = nbatches

    def __len__(self):
        "returns the number of batches to cover all the data"
        return np.sum([b for b in self.batches_per_ddid.values()])

    def __iter__(self):
        if self.shuffle:
            rng = np.random.default_rng()
            rng.shuffle(self.unique_ddids)

        for ddid in self.unique_ddids:
            # find all indices with ddid match
            ids_ddid = self.ids[(self.ddids == ddid)]

            for batch in torch.chunk(torch.tensor(ids_ddid), self.batches_per_ddid[ddid]):
                yield batch.tolist()


def plots(model, step, writer):
    """
    Plot images to the Tensorboard instance.
    """

    r = 1
    img = np.squeeze(utils.torch2npy(model.icube.sky_cube))
    fig, ax = plt.subplots(nrows=1)
    plot.plot_image(img, extent=model.coords.img_ext, ax=ax)
    # set zoom a little
    ax.set_xlim(r, -r)
    ax.set_ylim(-r, r)
    writer.add_figure("image", fig, step)

    norm_asinh = plot.get_image_cmap_norm(img, stretch='asinh')
    fig, ax = plt.subplots(nrows=1)
    plot.plot_image(img, extent=model.coords.img_ext, norm=norm_asinh, ax=ax)
    # set zoom a little
    ax.set_xlim(r, -r)
    ax.set_ylim(-r, r)
    writer.add_figure("asinh", fig, step)

    bcube = np.squeeze(utils.torch2npy(utils.packed_cube_to_sky_cube(model.bcube.base_cube)))
    norm = mco.Normalize(vmin=np.min(bcube), vmax=np.max(bcube))
    fig, ax = plt.subplots(nrows=1)
    plot.plot_image(bcube, extent=model.coords.img_ext, ax=ax, norm=norm)
    writer.add_figure("bcube", fig, step)

    # get gradient as it exists on model from root node
    b_grad = np.squeeze(utils.torch2npy(utils.packed_cube_to_sky_cube(model.bcube.base_cube.grad)))
    norm_sym = plot.get_image_cmap_norm(b_grad, symmetric=True)
    fig, ax = plt.subplots(nrows=1)
    plot.plot_image(b_grad, extent=model.coords.img_ext, norm=norm_sym, ax=ax, cmap="bwr_r")
    writer.add_figure("b_grad", fig, step)


def residual_dirty_image(coords, model_vis, uu, vv, data, weight, step, writer):
    # calculate residual dirty image for this *batch*
    resid = data - model_vis
    imager = gridding.DirtyImager.from_tensors(
        coords=coords, uu=uu, vv=vv, weight=weight, data=resid
    )
    img, beam = imager.get_dirty_image(
        weighting="briggs",
        robust=0.0,
        check_visibility_scatter=False,
        unit="Jy/arcsec^2",
    )

    fig = dirty_image.plot_beam_and_image(beam, img, imager.coords.img_ext)
    writer.add_figure("dirty_image", fig, step)

# plot histogram of residuals normalized to weight
def plot_residual_histogram(model_vis, data, weight, ddid, step, writer):
    # convert all quantities to numpy arrays 
    model_vis = utils.torch2npy(torch.squeeze(model_vis))
    data = utils.torch2npy(torch.squeeze(data))
    weight = utils.torch2npy(torch.squeeze(weight))

    ddid = utils.torch2npy(ddid)
    unique_ddid = np.unique(ddid)

    # compute normalized scatter 
    scatter = visread.scatter.get_averaged_scatter(data, model_vis, weight)
    fig = visread.visualization.plot_averaged_scatter(scatter)
    fig.suptitle(unique_ddid)

    writer.add_figure("residual_scatter", fig, step)

def train(args, model, device, train_loader, optimizer, epoch, lam_ent, writer):
    model.train()
    for i_batch, (uu, vv, data, weight, ddid) in enumerate(train_loader):
        # send all values to device
        uu, vv, data, weight, ddid = (
            uu.to(device),
            vv.to(device),
            data.to(device),
            weight.to(device),
            ddid.to(device)
        )

        optimizer.zero_grad()
        
        # get model visibilities
        vis = model(uu, vv)[0] # take only the first channel

        # correct the weights for the per-ddid weight rescale factors
        # these originated from bad pipeline calibration
        weight_adjusted = model.adjust_weights(ddid, weight)

        # allow for per-obsid adjustments of amplitude scaling
        vis_scaled, weight_scaled = model.adjust_amplitudes(ddid, vis, weight_adjusted)

        # allow for astrometric offsets on per obsid basis 
        vis_shifted = model.adjust_positions(ddid, uu, vv, vis_scaled)

        loss = losses.neg_log_likelihood_avg(
            vis_shifted, data, weight_scaled
        ) + lam_ent * losses.entropy(
            model.icube.packed_cube, prior_intensity=1e-4, tot_flux=0.253
        )

        loss.backward()
        optimizer.step()
        
        # log results
        if i_batch % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    i_batch * len(data),
                    len(train_loader.dataset),
                    100.0 * i_batch / len(train_loader),
                    loss.item(),
                )
            )

            step = i_batch + epoch * len(train_loader)
            writer.add_scalar("loss", loss.item(), step)
            
            sigma_dict = {str(ddid):10**val.item() for ddid, val in enumerate(model.log10_sigma_factors)}
            writer.add_scalars("sigma_scale_factors", sigma_dict, step)

            amp_dict = {str(obsid):10**val.item() for obsid, val in enumerate(model.log10_amp_factors)}
            writer.add_scalars("amp_scale_factors", amp_dict, step)

            ra_dict = {str(obsid):val.item() for obsid, val in enumerate(model.delta_ra)}
            writer.add_scalars("ra_dict", ra_dict, step)

            dec_dict = {str(obsid):val.item() for obsid, val in enumerate(model.delta_dec)}
            writer.add_scalars("dec_dict", dec_dict, step)

            ddid_str = "{:}".format(torch.unique(ddid).detach().cpu())
            writer.add_text("ddid", ddid_str, step)

            plots(model, step, writer)
            residual_dirty_image(model.coords, vis_shifted, uu, vv, data, weight_scaled, step, writer)
            plot_residual_histogram(vis_shifted, data, weight_scaled, ddid, step, writer)
            if args.dry_run:
                break

def main():
    # Training settings
    parser = argparse.ArgumentParser(description="IM Lup SGD Example")
    parser.add_argument(
        "asdf", help="Input path to .asdf file containing visibilities."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=40000,
        help="input batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-1,
        help="learning rate",
    )
    parser.add_argument("--FWHM", type=float, default=0.05, help="FWHM of Gaussian Base layer in arcseconds.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--tensorboard-log-dir",
        help="The log dir to which tensorboard files should be written.",
    )
    parser.add_argument("--lam-ent", type=float, default=0.0)
    parser.add_argument("--load-checkpoint", help="Path to checkpoint from which to resume.")
    parser.add_argument(
        "--save-checkpoint",
        help="Path to which checkpoint where finished model and optimizer state should be saved.",
    )
    parser.add_argument("--sampler", choices=["default", "ddid"], default="default")
    parser.add_argument("--sb_only", action="store_true", default=False)
    parser.add_argument("--freeze_weights", action="store_true")
    parser.add_argument("--freeze_amps", action="store_true")
    parser.add_argument("--freeze_positions", action="store_true")
    args = parser.parse_args()

    # set seed
    torch.manual_seed(args.seed)

    # choose the compute device, preference cuda > mps > cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # load the full dataset
    uu, vv, data, weight, ddid = loaddata.get_ddid_data(args.asdf, args.sb_only)

    # convert to float32
    uu = np.float32(uu)
    vv = np.float32(vv)
    data = np.complex64(data)
    weight = np.float32(weight)
    
    train_dataset = TensorDataset(
            torch.tensor(uu),
            torch.tensor(vv),
            torch.tensor(data),
            torch.tensor(weight),
            torch.tensor(ddid)
        )

    # set the batch sizes for the loaders
    cuda_kwargs = {"num_workers": 1, "pin_memory": True}
    
    if args.sampler == "default":
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **cuda_kwargs)

    elif args.sampler == "ddid":
        train_ddids = ddid
        train_sampler = DdidBatchSampler(train_ddids, args.batch_size)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, **cuda_kwargs)

    # create the model and send to device
    obsid_dict = {0:[0,1,2,3,4], 1:[5,6,7,8,9,10], 2:[11,12], 3:[13,14], 4:[15,16,17], 5:[18,19,20,21], 6:[22,23,24,25]}
    coords = coordinates.GridCoords(cell_size=0.005, npix=1028)

    model = Net(coords, FWHM=args.FWHM, obsid_dict=obsid_dict, freeze_amps=args.freeze_amps, freeze_weights=args.freeze_weights, freeze_positions=args.freeze_positions).to(device)

    # create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # here is where we could set up a scheduler, if desired

    if args.load_checkpoint is not None:
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # set up TensorBoard instance
    writer = SummaryWriter(log_dir=args.tensorboard_log_dir)

    # enter the loop
    for epoch in range(0, args.epochs):
        train(args, model, device, train_loader, optimizer, epoch, args.lam_ent, writer)
        optimizer.step()

        # overwrite checkpoint after each epoch
        if args.save_checkpoint is not None:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                },
                args.save_checkpoint,
            )

    writer.close()

if __name__ == "__main__":
    main()
