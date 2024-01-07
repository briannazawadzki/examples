# Getting started with Stochastic Gradient Descent (SGD)

This example folder collects several scripts that comprise a typical workflow. All scripts expect that you run them from this directory.

* `loaddata.py` generates mock visibilities on-the-fly from a source image and realistic baselines (stored in `data/mock_data.npy`).
* data exploration routines: `src/dirty_image.py` and `src/plot_baselines.py`. 
* RML imaging: `src/sgd.py`

# Usage 
You can produce the final image using Snakemake by
```
snakemake -c1 all
```

To experiment, we recommend an iterative workflow, save progress, and visualize your results as you explore the dataset, use commands like

```
python src/sgd.py --tensorboard-log-dir=runs/exp0 --save-checkpoint=checkpoints/0.pt 
```

More options with `python src/sgd.py --help`. View training progress with Tensorboard.

# Strategy

## Validation 
Since we are using mock data, we have the advantage of knowing the true sky. This allows us to calculate a 'validation loss' between the synthesized image and the true sky.

$$
L_\mathrm{validation} = \frac{1}{N} \sum_i^N (I_{\mathrm{true},i} - I_{\mathrm{syn}, i})^2
$$

This approach cannot be used with real datasets, obviously, but in this case allows us to precisely quantify the performance of each workflow configuration relaltive to each other. 

More precisely, we might calculate the validation score at several different resolutions, and monitor each of those.

## (lack of) Regularization
To demonstrate why regularization (beyond non-negative pixels) is needed for imaging workflows, try running without any:

```
python src/sgd.py --tensorboard-log-dir=runs/no-reg --save-checkpoint=checkpoints/6.py --epochs=10 --batch-size=1000
```

If run to convergence, you'll find a classic case of overfitting to the lower S/N visibilities at longer baselines / higher spatial frequencies. This manifests in the image as small splotches and/or individual pixels with very high flux concentrations. 

You can spot this behavior by monitoring the training loss and the validation loss with iteration. You begin to see the classic textbook signature of overfitting: the training loss monotonically decreases, while the validation loss first decreases but then starts to increase as the dataset is overfit.

## Maximum Entropy Regularization

* no regularization gave minimum validation loss of 7e-5 validation score (mean).
* ent 1e-5. reached minimum validation 7.7e-5, however still some residual flux in dirty image. Possibly unconverged.
* ent 1e-6. min validation 6.5e-5. loss is turning back up. Still some residual flux in dirty image, though. If you let it keep going, residual flux *mostly* disappears, though still some. Validation loss keeps diverging.
* try ent 5e-6: looking good, min val 5.96e-5 and still declining, many continued iterations. some residual flux remains. Continues to look good!

## Problems
* read text on optimizers - is Adam the best/simplest?
* Is it impossible to perfectly separate signal from noise here? To get a very good image (w/o noise corruption), we need to leave some flux in the residuals?
* Q: how widely applicable are the regularizer settings to other image morphologies? (same data properties)?
