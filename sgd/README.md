# Getting started with Stochastic Gradient Descent (SGD)

This example collects a few scripts that might be part of a typical workflow. The scripts expect that you run them from this directory.

`loaddata.py` uses the mock image and baselines produced by `generate-fake-data/` to create fake visibility data. 

Data exploration routines are in `src/dirty_image.py` and `src/plot_baselines.py`. 

The RML imaging routine is `src/sgd.py`. Can be run in an iterative manner, if you save progress.

You can optionally view training progress with tensorboard.

You can produce the final image using Snakemake by
```
snakemake -c1 all
```

![Baselines](analysis/baselines.png)
![Dirty Image](analysis/dirty_image.png)


