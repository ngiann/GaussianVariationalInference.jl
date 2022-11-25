## Specifying gradient options

Function `VI` allows the user to obtain a Gaussian approximation with minimal requirements. The user only needs to code a function `logp` that implements the log-posterior, provide an initial starting point `x₀` and call:

```
# log-posterior is a Gaussian with zero mean and unit covariance.
# Hence, our approximation should be exact in this example.
logp(x) = -sum(x.*x) / 2

# initial point implicitly specifies that the log-posterior is 5-dimensional
x₀ = randn(5)

# obtain approximation
q, logev = VI(logp, x₀, S = 200, iterations = 10_000, show_every = 200)

# Check that mean is close to zero and covariance close to identity.
# mean and cov are re-exported function from Distributions.jl
mean(q)
cov(q)
```
However, providing a gradient for `logp` can speed up the computation in `VI`.


#####  ➤  Gradient free mode

*Specify by* `gradientmode = :gradientfree`.

If no options relating to the gradient are specified, i.e. none of the options `gradientmode` or `gradlogp` is specified, `VI` will by default use internally the [`Optim.NelderMead`](https://julianlsolvers.github.io/Optim.jl/stable/#algo/nelder_mead/) optimiser that does not need a gradient.  

The user can explicitly specify that `VI` should use the gradient free optimisation algorithm  `Optim.NelderMead` by setting `gradientmode = :gradientfree`.



#####  ➤  Automatic differentiation mode

*Specify by* `gradientmode = :forward`.

If `logp` is coding a differentiable function[^1], then its gradient can be conveniently computed using automatic differentiation. By specifying `gradientmode = :forward`, function `VI` will internally use [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) to calculate the gradient of `logp`. In this
case, `VI` will use internally the [`Optim.LBFGS`](https://julianlsolvers.github.io/Optim.jl/stable/#algo/lbfgs/) optimiser.

```
q, logev = VI(logp, x₀, S = 200, iterations = 30, show_every = 1, gradientmode = :forward)
```

We note that with the use of `gradientmode = :forward` we arrive in fewer iterations to a result than in the gradient free case.


#####  ➤  Gradient provided

*Specify by* `gradientmode = :provided`.

The user can provide a gradient for `logp` via the `gradlogp` option:
```
# Let us calculate the gradient explicitly
gradlogp(x) = -x

q, logev = VI(logp, x₀, gradlogp = gradlogp, S = 200, iterations = 30, show_every = 1, gradientmode = :provided)
```

In this case, `VI` will use internally the [`Optim.LBFGS`](https://julianlsolvers.github.io/Optim.jl/stable/#algo/lbfgs/) optimiser. Again in this case we arrive in fewer iterations to a result than in the gradient free case.


!!! note

    Even if a gradient has been explicitly provided via the `gradlogp` option, the user still needs to specify `gradientmode = :provided` to instruct `VI` to use the provided gradient.




## Evaluating the lower bound on test samples

The options `S` specifies the number of samples to use when approximating the expected lower bound (ELBO), see [Technical background](@ref). The higher the value we use for `S`, the better the approximation to the ELBO will be, but at a higher computational cost. The lower the value we use for `S`, the faster the computation will be, but the approximation to the ELBO may be poorer. Hence, when setting `S` we need to take this trade-off into account.


Function `VI` offers a mechanism that tests whether the value `S` is set to a sufficiently high value. This mechanism makes use of two options, namely `Stest` and `test_every`. Option `Stest` defines a number of test samples used exclusively for evaluating (*not optimising!*) and reporting the ELBO every `test_every` number of iterations, see [ELBO maximisation](@ref).  If `S` is set sufficiently high, then we should see that as the ELBO increases, so does the ELBO on the test samples. If on the other hand, we notice that the ELBO on the test samples is decreasing, then this is a clear sign that a larger ``S`` is required.

Monitoring the ELBO this way is an effective way of detecting whether `S` has been set sufficiently high.

The following code snippet shows how to specify the options `Stest` and `test_every`:
```
# Use 2000 test samples and report test ELBO every 20 iterations
q, logev = VI(logp, x₀, S = 200, iterations = 3000, Stest = 2000, test_every = 20)
```

If the test ELBO at the current iteration is small than in the previous iteration, it is printed out in red colour. 

!!! note

    Whenever option `Stest` is set, `test_every` must be set to.

    
[^1]:The implementation of the function needs to satisfy certain requirements, see [here](https://juliadiff.org/ForwardDiff.jl/stable/user/limitations/).