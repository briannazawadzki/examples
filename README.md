# MPoL Examples

This repository hosts self-contained examples demonstrating [MPoL](https://mpol-dev.github.io/MPoL/) functionality. More info on each example can be found in the README.md within each example folder.

This repository is *not* continuously integrated with the rest of the codebase, because the computational demands are too significant. If you do encounter an error, please log it as a [GitHub issue](https://github.com/MPoL-dev/examples/issues).

## Getting started
* [Setup Mock Image and Baselines](generate-fake-data/README.md) | Generate a mock sky image $I_\nu(l,m)$ and interferometer baselines $(u,v)$ (but not visibilities $\mathcal{V}(u,v)$). These products are used as input for the other examples.
* [Stochastic Gradient Descent](sgd/README.md) | We recommend starting here for a complete example using MPoL to work with mock data.

## Advanced
* [Visibility Inference with Pyro](AS209-pyro-inference/README.md) | Use MPoL with Pyro to sample parametric visibility plane models.

## Stubs in progress
* [Diagnostic Imaging](diagnostic/gridder.md)
* [Optimization with Gridded Data](gridded/optimization.md)
* [Initialize to Dirty Image](dirty-image-initialization/initializedirtyimage.md)
* [Cross validation](crossvalidate/crossvalidation.md)