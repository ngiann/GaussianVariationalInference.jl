## Infer posterior of GP hyperparameters

In the following we approximate the intractable posterior of the hyperparameters of a Gaussian process. In order to reproduce this example, certain packages need to be independently installed.


```
using GaussianVariationalInference, Printf
using AbstractGPs, PyPlot, LinearAlgebra # These packages need to be independently installed


# Define log-posterior assuming a flat prior over hyperparameters

function logp(θ; x=x, y=y)

    local kernel = exp(θ[2]) * (Matern52Kernel() ∘ ScaleTransform(exp(θ[1])))

    local f = GP(kernel)

    local fx = f(x, exp(θ[3]))

    logpdf(fx, y)

end


# Generate some synthetic data

N = 25  # number of data items

σ = 0.2 # standard deviation of Gaussian noise

x = rand(N)*10  # ranomly sample 1-dimensional inputs

y = sin.(x) .+ randn(N)*σ # produce noise-corrupted outputs

xtest = collect(-1.0:0.1:11.0) # test inputs

ytest = sin.(xtest) # test outputs



# Approximate posterior with Gaussian

q, = VI(θ -> logp(θ; x=x, y=y), randn(3)*2, S = 300, iterations = 1000, show_every = 10)


# Draw samples from posterior and plot

for i in 1:3

    # draw hyperparameter sample from approximating Gaussian distribution
    local θ = rand(q)

    # instantiate kernel
    local sample_kernel = exp(θ[2]) * (Matern52Kernel() ∘ ScaleTransform(exp(θ[1])))

    # intantiate kernel, GP object and calculate posterior mean and covariance for the training data x, y generated above
    local f = GP(sample_kernel)
    local p_fx = AbstractGPs.posterior(f(x, exp(θ[3])), y)
    local μ, Σ = AbstractGPs.mean_and_cov(p_fx, xtest)

    figure()
    plot(x, y, "ko",label="Training data")
    plot(xtest, ytest, "b-", label="True curve")

    plot(xtest, μ, "r-")
    fill_between(xtest, μ.-sqrt.(diag(Σ)),μ.+sqrt.(diag(Σ)), color="r", alpha=0.3)

    title(@sprintf("GP posterior, sampled hyperparameters %.2f, %.2f, %.2f", exp(θ[1]),exp(θ[2]),exp(θ[3])))
    legend()

end
```


## Monitoring ELBO

We use again as our target distribution an unnormalised Gaussian.
```
using GaussianVariationalInference

# instantiate a covariance matrix
A = 0.1*randn(30, 30); Σ = A*A'

logp(x) = -sum(x'*(Σ\x)) / 2

# initial point
x₀ = randn(30)
```

We set `S=100` which is a quite low number of samples for inferring a 30-dimensional posterior. To diagnose whether `S` is set sufficiently high, we also test the ELBO on an indepedent set of samples of size `Stest=3000` every `test_every=20` iterations:

```
qlow, = VI(logp, x₀, S = 100, Stest = 3000, test_every = 10, iterations = 1000, gradientmode = :forward)
```

During execution we should see that there is a considerable gap between the ELBO and the test ELBO and that the test ELBO does not improve much. We now set `S=1000`:

```
qhigh, = VI(logp, x₀, S = 1000, Stest = 3000, test_every = 10, iterations = 1000, gradientmode = :forward)
```

In this case, we should see during execution that the reported ELBO is much closer to the test ELBO and the latter shows improvement. The improvement should also be visible in that the covariance of `qhigh` is closer to the true covariance:

```
norm(cov(qlow)  - Σ)    # this should be higher than value below
norm(cov(qhigh) - Σ)    # this should be lower  than value above
```

!!! note

    The results of this example depend strongly on the covariance ``\Sigma`` that we happened to sample.