# Getting started with Stochastic Gradient Descent (SGD)

Begins with the mock image and baselines produced by `generate-fake-data/`. 

* We want to take the image and baselines and, use the MPoL.fourier, produce mock visibilities, and treat this as a dataset.
    * `load_data.py` for generating things
    * `plot_baselines.py` 
    * `dirty_image.py` for checks
* Then, we will run an MPoL loop in an SGD pattern to see what we can get, without validation.
    * `sdg.py`
* Then, we will experiment validating against the true image, and see how/if we can improve the recovered image using regularization.


You can optionally view training progress with tensorboard.