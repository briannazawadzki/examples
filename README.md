# MPoL Examples

This repository hosts self-contained examples demonstrating [MPoL](https://mpol-dev.github.io/MPoL/) functionality. More info on each example can be found in the README.md within each example folder.

This repository is *not* continuously integrated with the rest of the codebase, because the computational demands are too significant. If you do encounter an error, please log it as a [GitHub issue](https://github.com/MPoL-dev/examples/issues).

## Getting started
* [01 - Setup Mock Image and Baselines](01-generate-mock-baselines/README.md) | Generate a mock sky image $I_\nu(l,m)$ and interferometer baselines $(u,v)$ (but not visibilities $\mathcal{V}(u,v)$). These products are used as input for the other examples.
* [02 - Stochastic Gradient Descent](02-sgd/README.md) | A complete end-to-end example using MPoL to image mock data.

## Advanced
* [03 - Visibility Inference with Pyro](03-AS209-pyro-inference/README.md) | Use MPoL with Pyro to sample parametric visibility plane models.
* [04 - IM Lup protoplanetary disk](04-IMLup-multi-EB) | Use MPoL to image the ALMA DSHARP observations of the IM Lup protoplanetary disk, taking into account alignment and weight-scaling adjustments for a multi-execution block dataset.

## Stubs in progress
* [Diagnostic Imaging](diagnostic/gridder.md)
* [Optimization with Gridded Data](gridded/optimization.md)
* [Initialize to Dirty Image](dirty-image-initialization/initializedirtyimage.md)
* [Cross validation](crossvalidate/crossvalidation.md)