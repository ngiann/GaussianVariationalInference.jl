# ApproximateVI.jl

This package implements approximate variational inference as presented in  
*Approximate variational inference based on a finite sample of Gaussian latent variables,  
Pattern Analysis and Applications volume 19, pages 475–485, 2015* [[DOI]](https://doi.org/10.1007/s10044-015-0496-9), [[Arxiv]](https://arxiv.org/pdf/1906.04507.pdf).

**This work in progress, documentation and more functionality will be added soon**


## What is this package about

This package implements variational inference using the re-parametrisation trick.
The work was published in the above [publication](https://arxiv.org/pdf/1906.04507.pdf). 
Of course the method has been popularised by the works ...


## What does the package do

The package offer function `VI`. This function approximates the posterior parameter distribution
with a Gaussian q(θ) = 𝜨(θ|μ,Σ) by minimizing the expected lower bound:

∫ q(θ) log p(x,θ) dθ + ℋ[q]

The above integral is approximated with a monte carlo average of S samples:

1/S log p(x,θₛ) dθ + ℋ[q]

Using the reparametrisation trick, we re-introduce the variational parameters that we need top optimise:

1/S log p(x,μ + √Σ zₛ) dθ + ℋ[q]

where √Σ is a matrix root of Σ (e.g. Cholesky "root") and zₛ∼𝜨(0,I).

A difference to other expositions, is the fact
that instead of repeatedly drawing new samples zₛ at each iteration of the optimiser, here a large number of samples zₛ is drawn
and kept fixed throughout the execution of the algorithm (see [paper](https://arxiv.org/pdf/1906.04507.pdf).
This avoids the difficulties of working with a noisy gradient and use optimisers like LBFGS, at the expense of risking overfitting to the samples zₛ that happen to be chosen. A mechanism for monitoring potential overfitting is described in the [paper](https://arxiv.org/pdf/1906.04507.pdf). Because of fixing the sample, the algorithm doesn't not scale well to high number of parameters and is thus recommended for problems with relatively few parameters, e.g. 2-20 parameters. Future work may address this limitation.


## How to use the package

The package is fairly easy to use. The only function relevant to the user is called `VI`.


## Examples

### Fitting a power law

### Integrating out hyperparameters in a Gaussian process

#### Monitoring "overfitting"

