# Technical description


## Variational inference via the re-parametrisation trick

Our goal is to approximate the true (unnormalised) posterior  distribution ``p(\theta|\mathcal{D})`` with a Gaussian ``q(\theta) = \mathcal{N}(\theta|\mu,\Sigma)`` by 
maximising the expected lower bound:

``\int q(\theta) \log p(x, \theta) d\theta + \mathcal{H}[q]``

also known as the ELBO. The above integral is approximated with as a Monte carlo average over ``S`` number of samples:

``\frac{1}{S} \sum_{s=1}^S \log p(x, \theta_s) + \mathcal{H}[q]``

Using the reparametrisation trick, we re-introduce the variational parameters that we need to optimise:

``\frac{1}{S} \sum_{s=1}^S \log p(x, \mu + C z_s) + \mathcal{H}[q]``

where ``z_s\sim\mathcal{N}(0,I)`` and ``C`` is a matrix root of ``\Sigma``, i.e. ``CC^T = \Sigma``.

By optimising the approximate lower bound with respect to the variational parameters ``\mu`` and ``C`` we obtain the approximate posterior ``q(\theta) = \mathcal{N}(\t
heta|\mu,CC^T)`` that offers the best Gaussian approximation to true posterior ``p(\theta|\mathcal{D})``.


## In more detail

This package implements variational inference using the re-parametrisation trick.
Contrary to other flavours of this method, that repeatedly draw new samples ``z_s`` at each iteration of the optimiser, here a large number of samples ``z_s`` is drawn
 at the start and is kept fixed throughout the execution of the algorithm[^1].
This avoids the difficulty of working with a noisy gradient and allows the use of optimisers like [LBFGS](https://julianlsolvers.github.io/Optim.jl/stable/#algo/lbfgs/
). A big advatange is that the use of LBFGS, does away with the typical requirement of tuning learning rates (step sizes). However, this comes at the expense of riskin
g overfitting to the samples ``z_s`` that happened to be drawn at the start. The package provides a mechanism for monitoring potential overfitting[^2] via the options 
`Stest` and `test_every`. Because of fixing the samples  ``z_s``, the algorithm doesn't not enjoy the speed of optimisation via stochastic gradient. As a consequence, 
the present package is recommented for problems with relatively few parameters, e.g. 2-20 parameters perhaps.


The work was independently developed and published [here](https://doi.org/10.1007/s10044-015-0496-9) [(Arxiv link)](https://arxiv.org/pdf/1906.04507.pdf).
Of course, the method has been widely popularised by the works [Doubly Stochastic Variational Bayes for non-Conjugate Inference](http://proceedings.mlr.press/v32/titsi
as14.pdf) and [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
The method indepedently appeared earlier in [Fixed-Form Variational Posterior Approximation through Stochastic Linear Regression](https://arxiv.org/abs/1206.6679) and 
later in [A comparison of variational approximations for fast inference in mixed logit models](https://link.springer.com/article/10.1007%2Fs00180-015-0638-y) and perha
ps in other publications too...


[^1]: See [paper](https://arxiv.org/pdf/1906.04507.pdf), Algorithm 1.
[^2]: See [paper](https://arxiv.org/pdf/1906.04507.pdf), Section 2.3.