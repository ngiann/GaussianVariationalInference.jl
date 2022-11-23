# ApproximateVI.jl


## What's this package for?

Approximate a posterior distribution with a Gaussian by minimising Kullback-Leibler divergence.



## Basic use

Currently, the main function of interest the package exposes is `VI`.
At the very minimum, the user needs to provide a function that codes the (unnormalised)log-posterior function.

Let's consider the following toy example:
```
using ApproximateVI

logp = exampleproblem1() # target log-posterior to approximate
x₀ = randn(2)            # random initial mean for approximating Gaussian
q, logev = VI(logp, x₀, S = 100, iterations = 100, show_every = 1)

```


Options `S` above specifies the number of samples to use in order to approximate the variational lower bound, i.e. the objective that which minimised produces the best Gaussian approximation. The higher `S` is set the better, however, at a higher computational cost. The lower `S` the faster the method, but the riskier to produce a biased solution, see [Technical description](@ref).

Let us plot the target posterior and the Gaussian approximation held in `q`:
```
using ApproximateVIUtilities # install this for auxiliary functionality

```



