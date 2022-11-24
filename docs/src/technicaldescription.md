# Technical description


## ELBO maximisation

Our goal is to approximate the true (unnormalised) posterior  distribution ``p(\theta|\mathcal{D})`` with a Gaussian ``q(\theta) = \mathcal{N}(\theta|\mu,\Sigma)`` by 
maximising the expected lower bound:

``\int q(\theta) \log p(x, \theta) d\theta + \mathcal{H}[q]``,

also known as the ELBO. The above integral is approximated with as a Monte carlo average over ``S`` number of samples:

``\frac{1}{S} \sum_{s=1}^S \log p(x, \theta_s) + \mathcal{H}[q]``.

Using the reparametrisation trick, we re-introduce the variational parameters that we need to optimise:

``\frac{1}{S} \sum_{s=1}^S \log p(x, \mu + C z_s) + \mathcal{H}[q]``,

where ``z_s\sim\mathcal{N}(0,I)`` and ``C`` is a matrix root of ``\Sigma``, i.e. ``CC^T = \Sigma``.

By maximising the approximate ELBO with respect to the variational parameters ``\mu`` and ``C`` we obtain the approximate posterior ``q(\theta) = \mathcal{N}(\theta|\mu,CC^T)`` that is the best Gaussian approximation to true posterior ``p(\theta|\mathcal{D})``.

The number of samples ``S`` in the above description, can be controlled via the option `S` when calling `VI`. 


## Monitoring ELBO on an independent test set of samples

This package implements variational inference using the re-parametrisation trick.
Contrary to other flavours of this method, that repeatedly draw ``S`` new samples ``z_s`` at each iteration of the optimiser, here we draw at the start  a large number ``S`` of samples ``z_s`` and keep them fixed throughout the execution of the algorithm[^1]. This avoids the difficulty of working with a noisy gradient and allows the use of optimisers like [LBFGS](https://julianlsolvers.github.io/Optim.jl/stable/#algo/lbfgs/). Using LBFGS, does away with the typical requirement of tuning learning rates (step sizes). However, this comes at the expense of risking overfitting to the samples ``z_s`` that happened to be drawn at the start. 

The package provides a mechanism for monitoring potential overfitting[^2] via the options `Stest` and `test_every`, see [Evaluating the lower bound on test samples](@ref). Indepedently of the samples ``z_s``, a test set of ``S^{(test)}`` number of samples, denoted as ``z_s^{(test)}``, is drawn and kept kept fixed. This second set of sample is used to periodically monitor the ELBO:

``\frac{1}{S^{(test)}} \sum_{s=1}^{S^{(test)}} \log p(x, \mu + C z_s^{(test)}) + \mathcal{H}[q]``,

Because of fixing the samples  ``z_s``, the algorithm doesn't not enjoy the same scalability as variational inference with stochastic gradient does. As a consequence, 
the present package is recommented for problems with relatively few parameters, e.g. 2-20 parameters perhaps.


The work was independently developed and published [here](https://doi.org/10.1007/s10044-015-0496-9) [(Arxiv link)](https://arxiv.org/pdf/1906.04507.pdf).
Of course, the method has been widely popularised by the works [Doubly Stochastic Variational Bayes for non-Conjugate Inference](http://proceedings.mlr.press/v32/titsi
as14.pdf) and [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
The method seems to have appeared earlier in [Fixed-Form Variational Posterior Approximation through Stochastic Linear Regression](https://arxiv.org/abs/1206.6679) and again later in [A comparison of variational approximations for fast inference in mixed logit models](https://link.springer.com/article/10.1007%2Fs00180-015-0638-y) and perhaps in other publications too...


[^1]: See [paper](https://arxiv.org/pdf/1906.04507.pdf), Algorithm 1.
[^2]: See [paper](https://arxiv.org/pdf/1906.04507.pdf), Section 2.3.