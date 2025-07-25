import numpy as np
from mpol import losses

def train(
    model, dataset, optimizer, scheduler, config, device, writer=None, report=False, logevery=50
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
    model = model.to(device)
    model.train()
    dataset = dataset.to(device)

    loss_tracker = []
    for iteration in range(config["epochs"]):

        optimizer.zero_grad()
        vis = model.forward()
        sky_cube = model.icube.sky_cube

        loss = (
            losses.nll_gridded(vis, dataset)
            + config["lambda_TV"] * losses.TV_image(sky_cube)
            + config["lambda_TSV"] * losses.TSV(sky_cube)
	        + config["lambda_sparsity"] * losses.sparsity(sky_cube)
	        + config["entropy"] * losses.entropy(sky_cube, config["prior_intensity"], config["tot_flux"])
        )

        loss_tracker.append(loss.item())
        if (iteration % logevery == 0) and writer is not None:
            writer.add_scalar("loss", loss.item(), iteration)

        loss.backward()
        optimizer.step()
        scheduler.step()

    if report:
        tune.report(loss=loss.item())

    return loss.item(), loss_tracker


def test(model, dataset, device):
    model = model.to(device)
    model.eval()
    dataset = dataset.to(device)
    vis = model.forward()
    loss = losses.nll_gridded(vis, dataset)
    return loss.item()

