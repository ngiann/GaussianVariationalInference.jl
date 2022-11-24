# More options

## Specifying gradient options

Function `VI` allows the user to obtain a Gaussian approximation with minimal requirements. 
The user only needs to code a function `logp` that implements the log-posterior, provide an initial starting point `x₀` and call:

```
# log-posterior is an unnormalised Gaussian,
# hence our approximation should be exact in this example.
logp(x) = -sum(x.*x) / 2

# implicitly specifies that the log-posterior is 5-dim
x₀ = randn(5)

# obtain approximation
q, logev = VI(logp, x₀, S = 200, iterations = 10_000, show_every = 200)
```
However, providing a gradient for `logp` can speed up the computation in `VI`.


#####  ➤  Gradient free mode

*Specify by* `gradientmode = :gradientfree`.

If no options relating to the gradient are specified, i.e. none of the options `gradientmode` or `gradlogp` is specified, `VI` will by default use internally the [`Optim.NelderMead`](https://julianlsolvers.github.io/Optim.jl/stable/#algo/nelder_mead/) optimiser that does not need a gradient.  

The user can explicitly specify that the algorithm should use the gradient free [`Optim.NelderMead`](https://julianlsolvers.github.io/Optim.jl/stable/#algo/nelder_mead/) optimisation algorithm by setting `gradientmode = :gradientfree`.



#####  ➤  Automatic differentiation mode

*Specify by* `gradientmode = :forward`.

If `logp` is coding a differentiable function, the its gradient can be conveniently computed using automatic differentiation. By specifying `gradientmode = :forward`, function `VI` will internally use [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) to calculate the gradient of `logp`. In this
case, `VI` will use internally the [`Optim.LBFGS`](https://julianlsolvers.github.io/Optim.jl/stable/#algo/lbfgs/) optimiser.

```
q, logev = VI(logp, x₀, S = 200, iterations = 30, show_every = 1, gradientmode = :forward)
```

We note that with the use of `gradientmode = :forward` we arrive in fewer iterations to a result than in the gradient free case.


#####  ➤  Gradient provided

*Specify by* `gradientmode = :provided`.

The user can provide a gradient for `logp` via the `gralogp` option:
```
# Let us calculate the gradient explicitly
gradlogp(x) = -x

q, logev = VI(logp, x₀, gradlogp = gradlogp, S = 200, iterations = 30, show_every = 1, gradientmode = :provided)
```

In this case, `VI` will use internally the [`Optim.LBFGS`](https://julianlsolvers.github.io/Optim.jl/stable/#algo/lbfgs/) optimiser. Again in this case we arrive in fewer iterations to a result than in the gradient free case.


!!! note

    Even if a gradient has been explicitly provided via the `gralogl` option, the user still needs to specify `gradientmode = :provided` to instruct `VI` to use the provided gradient.


## Evaluating the lower bound on test samples
