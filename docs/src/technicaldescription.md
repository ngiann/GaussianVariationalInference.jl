# Technical description

We provide details on implemented algorithm. In short, the algorithm maximises the variational lower bound, typically known as the ELBO, using the reparametrisation trick.


## ELBO maximisation

Our goal is to approximate the true (unnormalised) posterior  distribution ``p(\theta|\mathcal{D})`` with a Gaussian ``q(\theta) = \mathcal{N}(\theta|\mu,\Sigma)`` by 
maximising the expected lower bound:

``\mathcal{L}(\mu,\Sigma) = \int q(\theta) \log p(\mathcal{D}, \theta) d\theta + \mathcal{H}[q]``,

also known as the ELBO. The above integral will be in general intractable . We can make progress by approximating it with as a Monte carlo average over ``S`` number of samples ``\theta_s\sim q(\theta)``:

``\mathcal{L}(\mu,\Sigma) \approx \frac{1}{S} \sum_{s=1}^S \log p(\mathcal{D}, \theta_s) + \mathcal{H}[q]``.

Due to the sampling, however, the variational parameters no longer appear in the above approximation. Nevertheless, it is possible to re-introduce them by rewriting the sampled parameters as:

``\theta_s = \mu + C z_s``,


where ``z_s\sim\mathcal{N}(0,I)`` and ``C`` is a matrix root of ``\Sigma``, i.e. ``CC^T = \Sigma``.
We refer collectively to all samples as ``Z = \{z_1 . . . , z_S \}``.
This is known as the reparametrisation trick. Now we are able to we re-introduce the variational parameters in the approximated ELBO:

``\mathcal{L}_{(FS)}(\mu,C,Z) = \frac{1}{S} \sum_{s=1}^S \log p(\mathcal{D}, \mu + C z_s) + \mathcal{H}[q]``,

where the subscript ``FS`` stands for *finite sample*. We denote the approximate ELBO with ``\mathcal{L}_{(FS)}(\mu,C,Z)`` and make it explicit that it depends on the samples ``Z``. 
By maximising the approximate ELBO ``\mathcal{L}_{(FS)}(\mu,C,Z)`` with respect to the variational parameters ``\mu`` and ``C`` we obtain the approximate posterior ``q(\theta) = \mathcal{N}(\theta|\mu,CC^T)`` that is the best Gaussian approximation to true posterior ``p(\theta|\mathcal{D})``.



## Choosing the number of samples ``S``

For large values of ``S`` the proposed approximate lower bound ``\mathcal{L}_{(FS)}(\mu,C,Z)`` approximates the true bound ``\mathcal{L}(\mu,\Sigma)`` closely.
Therefore, we expect that optimising ``\mathcal{L}_{(FS)}(\mu,C,Z)`` will yield
approximately the same variational parameters ``µ, C`` as the optimisation of the intractable true lower bound ``\mathcal{L}(\mu,\Sigma)`` would.


Here the samples ``z_s`` are drawn at start of the algorithm and are kept fixed throughout its execution. The proposed scheme exhibits some fluctuation as ``\mathcal{L}_{(FS)}(\mu,C)`` depends on the random set of samples ``z_s`` that happened to be drawn at the start of the algorithm. Hence, for another set of randomly drawn samples ``z_s`` the function  ``\mathcal{L}_{(FS)}(\mu,C)`` will be (hopefully only slightly) different. However, for large enough ``S`` the fluctuation due to ``z_s`` should be innocuous and optimising it should yield approximately the same variational parameters for any drawn ``z_s``.


However, if on the other hand we choose a small value for ``S``, then the variational parameters will overly depend on the small set of samples ``z_s`` that happened to be drawn at the beginning of the algorithm. As a consequence, ``\mathcal{L}_{(FS)}(\mu,C,Z)`` will approximate ``\mathcal{L}(\mu,\Sigma)`` poorly, and the resulting posterior ``q(\theta)`` will also be a poor approximation to the true posterior. ``p(\theta|\mathcal{D})``. Hence, the variational parameters will be overadapted to the small set of samples ``z_s`` that happened to be drawn. Naturally, the question arises of how to choose a large enough ``S`` in order avoid the sitatuation where ``\mathcal{L}_{(FS)}(\mu,C,Z)`` over-adapts to  variational parameters to the samples ``z_s``. 




## Monitoring ELBO on an independent test set of samples

A practical answer to diagnosing whether a sufficiently high number of samples ``S`` has been chosen,is the following: at the beginning of the algorithm we draw a second independent set of samples ``Z^\prime =\{ z_1^\prime, z_2^\prime, \dots, z_S^\prime\}`` where ``S^\prime`` is preferably a number larger than ``S``. 

At each (or every few) iteration(s) we monitor the quantity ``\mathcal{L}_{(FS)}(\mu,C,Z^\prime)`` on the independent sample set ``Z^\prime``. If the variational parameters are not overadapting to the ``Z``, then we should see that as the lower bound ``\mathcal{L}_{(FS)}(\mu,C,Z)``  increases, the quantity ``\mathcal{L}_{(FS)}(\mu,C,Z^\prime)``  should also display a tendency to increase. If on the other hand the variational parameters are overfitting the drawn
Z, then though ``\mathcal{L}_{(FS)}(\mu,C,Z)`` is increasing, we
will notice that ``\mathcal{L}_{(FS)}(\mu,C,Z^\prime)``  is actually deteriorating. This is a clear sign that a larger ``S`` is required.

The described procedure is reminiscent of monitoring the generalisation performance of a learning algorithm on a validation set during training. A significant difference, however, is that while validation sets are typically of limited size, here we can set ``S^\prime`` arbitrarily large. In practice, one may experiment with such values as e.g. ``S^\prime = 2S`` or  ``S^\prime = 10S``.



## Relevant options in `VI`

This package implements variational inference using the re-parametrisation trick.
Contrary to other flavours of this method, that repeatedly draw ``S`` new samples ``z_s`` at each iteration of the optimiser, here we draw at the start  a large number ``S`` of samples ``z_s`` and keep them fixed throughout the execution of the algorithm[^1]. This avoids the difficulty of working with a noisy gradient and allows the use of optimisers like [LBFGS](https://julianlsolvers.github.io/Optim.jl/stable/#algo/lbfgs/). Using LBFGS, does away with the typical requirement of tuning learning rates (step sizes). However, this comes at the expense of risking overfitting to the samples ``z_s`` that happened to be drawn at the start. Because of fixing the samples  ``z_s``, the algorithm doesn't not enjoy the same scalability as variational inference with stochastic gradient does. As a consequence, 
the present package is recommented for problems with relatively few parameters, e.g. 2-20 parameters perhaps.


As explained in the previous section, one may monitor the approximate ELBO on an independent set of samples ``Z^\prime`` of size ``S^\prime``. The package provides a mechanism for monitoring potential overfitting[^2] via the options `Stest` and `test_every`. Options `Stest` set the number of test samples ``S^\prime`` and `test_every` specifies how often we should monitor the approximate ELBO on ``Z^\prime``
by evaluating ``\mathcal{L}_{(FS)}(\mu,C,Z^\prime)``.

!!! note

    Whenever option `Stest` is set, `test_every` must be set to.




## Literature

The work was independently developed and published [here](https://doi.org/10.1007/s10044-015-0496-9) [(Arxiv link)](https://arxiv.org/pdf/1906.04507.pdf).
Of course, the method has been widely popularised by the works [Doubly Stochastic Variational Bayes for non-Conjugate Inference](http://proceedings.mlr.press/v32/titsi
as14.pdf) and [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
The method seems to have appeared earlier in [Fixed-Form Variational Posterior Approximation through Stochastic Linear Regression](https://arxiv.org/abs/1206.6679) and again later in [A comparison of variational approximations for fast inference in mixed logit models](https://link.springer.com/article/10.1007%2Fs00180-015-0638-y) and perhaps in other publications too...


[^1]: See [paper](https://arxiv.org/pdf/1906.04507.pdf), Algorithm 1.
[^2]: See [paper](https://arxiv.org/pdf/1906.04507.pdf), Section 2.3.