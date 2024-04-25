# SGD Gauss

This is an introductory workflow to explain how MPoL works, using simulated data.

It has a Gaussian convolution layer at the base.

# Validation 
Since we are using mock data, we have the advantage of knowing the true sky. This allows us to calculate a 'validation loss' between the synthesized image and the true sky.

$$
L_\mathrm{validation} = \frac{1}{N} \sum_i^N (I_{\mathrm{true},i} - I_{\mathrm{syn}, i})^2
$$

This approach cannot be used with real datasets, obviously, but in this case affords many benefits to precisely quantify the performance of each workflow configuration.

## Varying resolution

If the dataset lacks many long baselines, it is unrealistic for RML to recover the native resolution of the image. In this case, we can calculate the validation score at resolutions coarser than the source image. We do this by convolving both $I_\mathrm{true}$ and $I_\mathrm{syn}$ with a 2D Gaussian described by FWHMs of $\theta_a, \theta_b$  before computing $L_\mathrm{validation}$. 

# (lack of) Regularization
To demonstrate why regularization is needed for imaging workflows, try running without any:

```
python src/sgd.py --tensorboard-log-dir=runs/nolam0 --epochs=40 --log-interval=2 --save-checkpoint=checkpoints/nolam0.pt --lr 1e-2
```

If run to convergence, you'll find a classic case of overfitting to the lower S/N visibilities at longer baselines / higher spatial frequencies. This manifests in the image as small splotches and/or individual pixels with very high flux concentrations. If we didn't enforce non-negative pixels by construction, this would probably manifest as high frequency "noise" similar to uniformly-weighted images.

You can spot this behavior by monitoring the training loss and the validation loss with iteration. You will see the classic textbook signature of overfitting: the validation loss decreases for a while but eventually turns around and increases, while the training loss monotonically decreases as it fits the signal and then eventually tries to fit all the noise. One could attempt to regularize this behavior away using early stopping. However, in practice with real data we would not have access to a validation, so we look to alternative regularization techniques.

# Maximum Entropy Regularization

Things we can vary:
* fixed FWHM setting of `GaussBaseBeam`
* amount of max ent regularization

Things we record:
* validation loss at several final resolutions.


FWHM = 0.05, no entropy regularization
Raw is 4.8e-5
0.02 is 3.2e-5
0.06 is 1.16e-5
0.1 is 4.8e-6
over train and runaway fits

FWHM = 0.05, lam = 1e-4 entropy


Visualization
* compare known visibility amplitude to measured, as a function of radial baseline