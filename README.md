<h1 align="center">ApproximateVI.jl</h1>
<p align="center">
<img src="https://github.com/ngiann/ApproximateVI.jl/blob/4d604d2f42f74c97a84685ddf13e0a9d05ff76e7/docs/src/assets/logo.png" width="192" height="144">
</p>

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Documentation](https://img.shields.io/badge/docs-master-blue.svg)](https://ngiann.github.io/ApproximateVI.jl)
![GitHub](https://img.shields.io/github/license/ngiann/approximateVI.jl)

# What is this?

A Julia package for approximating a posterior distribution with a full-covariance Gaussian distribution by optimising a variational lower bound[^1]. In the near future it is planned to introduce a method for mean-field approximation. We recommend using this package for problems with a relatively small number of parameters, 2-20 parameters perhaps.



## Basic usage

To install this package, please switch in the REPL into package mode and add using `add ApproximateVI`.

The package is fairly easy to use. Currently, the only function of interest to the user is `VI`. At the very minimum, the user needs to provide a function that codes the joint log-likelihood function.

Consider approximating the following target density:

```
using ApproximateVI

logp = exampleproblem1() # target log-posterior density to approximate
x₀ = randn(2)            # random initial mean for approximating Gaussian
q, logev = VI(logp, randn(2), S = 100, iterations = 10_000, show_every = 50)

# Plot target posterior, not log-posterior!
using Plots # must be indepedently installed.
x = -3:0.02:3
contour(x, x, map(x -> exp(logp(collect(x))), Iterators.product(x, x))', fill=true, c=:blues)

# Plot Gaussian approximation on top using red colour
contour!(x, x, map(x -> pdf(q,(collect(x))), Iterators.product(x, x))', color="red", alpha=0.2)
```

A plot similar to the one below should appear. The  blue filled contours correspond to the exponentiated `logp`, and the red contours correspond to the produced Gaussian approximation `q`.

![image](docs/src/exampleproblem1.png)

For further information, please consult the documentation.


## Related, useful packages

- [AdvancedVI.jl](https://github.com/TuringLang/AdvancedVI.jl): A library for variational Bayesian inference in Julia.
- [DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl): Implementation of robust dynamic Hamiltonian Monte Carlo methods in Julia.


[^1]:[Approximate Variational Inference Based on a Finite Sample of Gaussian Latent Variables](https://doi.org/10.1007/s10044-015-0496-9), [[Arxiv]](https://arxiv.org/pdf/1906.04507.pdf).
