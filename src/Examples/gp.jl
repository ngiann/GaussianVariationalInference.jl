#####################################################################
# Example: integrate out the hyperparameters of a Gaussian process. #
#####################################################################

# Extra packages needed for this example
# If missing, they must be installed explicitly!

using AbstractGPs, PyPlot, Printf, LinearAlgebra


# Define log-likelihood

function logp(θ; x=x, y=y)

    local kernel = exp(θ[2]) * (Matern52Kernel() ∘ ScaleTransform(exp(θ[1])))

    local f = GP(kernel)

    local fx = f(x, exp(θ[3]))

    logpdf(fx, y)

end


# Generate some synthetic data

N = 25

σ = 0.1

x = rand(N)*10

y = sin.(x) .+ randn(N)*σ

xtest = collect(-1.0:0.1:11.0)

ytest = sin.(xtest)


# Approximate posterior with Gaussian

postθ, = VI( θ ->  logp(θ; x=x, y=y), [randn(3)*2 for i=1:10], S = 200, iterations = 100, show_every = 10)


# Draw samples from posterior and plot

for i in 1:3

    # draw hyperparameter sample from approximating Gaussian distribution
    local θ = rand(postθ)

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
