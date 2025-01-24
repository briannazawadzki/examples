# Bayesian Inference Example

This tutorial walks through how one can use MPoL with the probabilistic programming language Pyro to do Bayesian inference with visibility datasets. We show how to use Stochastic Variational Inference (SVI), but Hamiltonian Monte Carlo samplers are also possible.

The "source" of the tutorial is the `pyro.md` file, and the output `.ipynb` file is generated via 

```
jupytext --to ipynb --execute pyro.md
```

However, if you're just following along, you can view the `pyro.ipynb` file directly, since it contains the figure outputs.