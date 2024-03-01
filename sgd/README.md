# Getting started with Stochastic Gradient Descent (SGD)

This example folder collects several scripts that comprise a typical workflow. All scripts expect that you run them from this directory.

* `loaddata.py` generates mock visibilities on-the-fly from a source image and realistic baselines (stored in `data/mock_data.npy`).
* data exploration routines: `src/dirty_image.py` and `src/plot_baselines.py`. 
* RML imaging: `src/sgd.py`

# Usage 
You can produce a final image using Snakemake by
```
snakemake -c1 all
```

To experiment, we recommend an iterative workflow where you save progress and visualize your results as you explore the dataset. For example, use commands like

```
python src/sgd.py --tensorboard-log-dir=runs/exp0 --save-checkpoint=checkpoints/0.pt 
```

and then view training progress with Tensorboard.

# Strategy

## Validation 
Since we are using mock data, we have the advantage of knowing the true sky. This allows us to calculate a 'validation loss' between the synthesized image and the true sky.

$$
L_\mathrm{validation} = \frac{1}{N} \sum_i^N (I_{\mathrm{true},i} - I_{\mathrm{syn}, i})^2
$$

This approach cannot be used with real datasets, obviously, but in this case affords many benefits to precisely quantify the performance of each workflow configuration.

### Varying resolution

If the dataset lacks many long baselines, it is unrealistic for RML to recover the native resolution of the image. In this case, we can calculate the validation score at resolutions coarser than the source image. We do this by convolving both $I_\mathrm{true}$ and $I_\mathrm{syn}$ with a 2D Gaussian described by FWHMs of $\theta_a, \theta_b$  before computing $L_\mathrm{validation}$. 

## (lack of) Regularization
To demonstrate why regularization is needed for imaging workflows, try running without any:

```
python src/sgd.py --tensorboard-log-dir=runs/nolam0 --epochs=40 --log-interval=2 --save-checkpoint=checkpoints/nolam0.pt --lr 1e-2
```

If run to convergence, you'll find a classic case of overfitting to the lower S/N visibilities at longer baselines / higher spatial frequencies. This manifests in the image as small splotches and/or individual pixels with very high flux concentrations. If we didn't enforce non-negative pixels by construction, this would probably manifest as high frequency "noise" similar to uniformly-weighted images.

You can spot this behavior by monitoring the training loss and the validation loss with iteration. You will see the classic textbook signature of overfitting: the validation loss decreases for a while but eventually turns around and increases, while the training loss monotonically decreases as it fits the signal and then eventually tries to fit all the noise.

## Maximum Entropy Regularization

* no regularization w/ SGD gave minimum validation loss of 7e-5 validation score (mean).
* no reg w/ AdamW gave min 8.3e-5
* ent 1e-5. reached minimum validation 7.7e-5, however still some residual flux in dirty image. Possibly unconverged.
* ent 1e-6. min validation 6.5e-5. loss is turning back up. Still some residual flux in dirty image, though. If you let it keep going, residual flux *mostly* disappears, though still some. Validation loss keeps diverging.
* try ent 5e-6: looking good, min val 5.96e-5 and still declining, many continued iterations. some residual flux remains. Continues to look good!

# Next steps
* calculate the validation score at several different resolutions, and monitor each of those as a tensorboard scalar
* Could also put taper as learnable parameter in feedforward (instead of Hann)
* read text on optimizers - should we be tuning AdamW somehow?
* Is it impossible to perfectly separate signal from noise here? To get a very good image (w/o noise corruption), we need to leave some flux in the residuals?
* Q: how widely applicable are the regularizer settings to other image morphologies? (same data properties)?
