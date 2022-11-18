"""
# Basic use:

    q, logev = VI(logl, μ, [σ²=0.1]; S = 100, iterations = 1, show_every = -1)

Returns approximate Gaussian posterior and log evidence

# Arguments

Arguments in brackets are optional.

- `logl` is a function that expresses the joint log-likelihood
- `μ` is the initial mean of the approximating Gaussian posterior.
- `σ²` specifies the initial covariance as σ² * I of the approximating Gaussian posterior. Default value is `0.1`.
- `S` is the number of drawn samples that approximate the lower bound integral.
- `show_every`: report progress every `show_every` number of iterations.

# Outputs

- `q` is the approximating posterior returned as a ```Distributions.MvNormal``` type
- `logev` is the approximate log-evidence.

More options are explained in the README.md file.

# Example

```julia-repl
# infer posterior of Bayesian linear regression, compare to exact result
julia> using LinearAlgebra, Distributions
julia> D = 4; X = randn(D, 1000); W = randn(D); β = 0.3; α = 1.0;
julia> Y = vec(W'*X); Y += randn(size(Y))/sqrt(β);
julia> Sn = inv(α*I + β*(X*X')) ; mn = β*Sn*X*Y; # exact posterior
julia> posterior, logev = VI( w -> logpdf(MvNormal(vec(w'*X), sqrt(1/β)), Y) + logpdf(MvNormal(zeros(D),sqrt(1/α)), w), randn(D); S = 1_000, iterations = 15);
julia> display([mean(posterior) mn])
julia> display([cov(posterior)  Sn])
julia> display(logev) # display negative log evidence
```

"""
function VI(logl::Function, μ::Array{T, 1}, Σ::Array{T, 2}; gradlogl = defaultgradient(μ), gradientmode = :forward, seed::Int = 1, S::Int = 100, iterations::Int=1, numerical_verification::Bool = false, Stest::Int = 0, show_every::Int = -1, test_every::Int = -1) where T


    # check validity of arguments

    @argcheck seed >0           

    @argcheck iterations > 0    
    
    @argcheck S > 0         
    
    @argcheck size(Σ, 1) == size(Σ, 2)  "Σ must be a square matrix"
    
    @argcheck isposdef(Σ)               "Σ must be positive definite"
    
    @argcheck length(μ) >= 2
    

    # check gradient arguments

    optimiser = NelderMead() # default optimiser


    if gradientmode == :forward
        
        gradlogl = x -> ForwardDiff.gradient(logl, x)

        optimiser = LBFGS() # optimiser to be used with gradient calculated wiht automatic differentiation

    elseif gradientmode == :provided

        if any(isnan.(gradlogl(μ)))
            
            error("provided gradient returns NaN when evaluate at provided μ")

        end

        optimiser = LBFGS() # optimiser to be used with user provided gradient

    elseif gradientmode == :gradientfree
        
        optimiser = NelderMead() # optimiser when no gradient provided

    else

        error("invalid specification of argument gradientmode")

    end


    # Call actual algorithm

    coreVIfull(logl, μ, Σ; gradlogl = gradlogl, seed = seed, S = S, optimiser=optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, test_every = test_every)

end


function VI(logl::Function, μ::Array{T, 1}, σ²::T = 0.1; gradlogl = defaultgradient(μ), gradientmode = :forward, seed::Int = 1, S::Int = 100, iterations::Int=1, numerical_verification::Bool = false, Stest::Int=0, show_every::Int = -1, test_every::Int = -1) where T

    @argcheck σ² > 0
    
    # initial covariance

    D = length(μ)
    
    Σ = Matrix(σ²*I, length(μ), length(μ))

    VI(logl, μ, Σ; gradlogl = gradlogl, gradientmode = gradientmode, seed = seed, S = S, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, test_every = test_every)

end


function VI(logl::Function, initgaussian::MvNormal; gradlogl = defaultgradient(mean(initgaussian)), gradientmode = :forward, seed::Int = 1, S::Int = 100, iterations::Int = 1, numerical_verification::Bool = false, Stest::Int = 0, show_every::Int = -1,  test_every::Int = -1)

    VI(logl, mean(initgaussian), cov(initgaussian); gradlogl = gradlogl, gradientmode = gradientmode, seed = seed, S = S, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, test_every = test_every)

end




# #-----------------------------------#
# # Call mean field                   #
# #-----------------------------------#

# function VIdiag(logl::Function, μ::Array{Float64,1}, Σdiag = 0.1*ones(length(μ)); gradlogl = x -> ForwardDiff.gradient(logl, x), optimiser=Optim.LBFGS(), seed = 1, S = 100, iterations=1, numerical_verification = false, Stest=0, show_every=-1, inititerations=0)

#     coreVIdiag(logl, [μ], [Σdiag]; gradlogl = gradlogl, seed = seed, S = S, optimiser = optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, inititerations=inititerations)

# end


# function VIdiag(logl::Function, initgaussian::MvNormal; gradlogl = x -> ForwardDiff.gradient(logl, x), optimiser=Optim.LBFGS(), seed = 1, S = 100, iterations=1, numerical_verification = false, Stest=0, show_every=-1, inititerations=0)

#     VIdiag(logl, mean(initgaussian), diag(cov(initgaussian)); gradlogl = gradlogl, seed = seed, S = S, optimiser = optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, inititerations=inititerations)

# end


# function VIdiag(logl::Function, μ::Array{Array{Float64,1},1}, Σdiag = [0.1*ones(length(μ[1])) for _ in 1:length(μ)]; gradlogl = x -> ForwardDiff.gradient(logl, x), optimiser=Optim.LBFGS(), seed = 1, S = 100,  iterations=1, numerical_verification = false, Stest=0, show_every=-1, inititerations=0)


#     coreVIdiag(logl, μ, Σdiag; gradlogl = gradlogl, seed = seed, S = S, optimiser = optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, inititerations=inititerations)


# end


# #-----------------------------------#
# # Call VI with spherical covariance #
# #-----------------------------------#

# function VIfixedcov(logl::Function, μ::Array{Float64,1}, fixedC::Array{Float64,2}; gradlogl = x -> ForwardDiff.gradient(logl, x), optimiser=Optim.LBFGS(), seed = 1, S = 100, iterations=1, numerical_verification = false, Stest=0, show_every=-1, inititerations=0, adaptvariance = 1)

#     coreVIfixedcov(logl, μ, fixedC, gradlogl = gradlogl, seed = seed, S = S, optimiser = optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, inititerations = inititerations, adaptvariance = adaptvariance)

# end


# #-----------------------------------#
# # Call Laplace                      #
# #-----------------------------------#

# function laplace(logl::Function, x::Array{Float64,1}; gradlogl = x -> ForwardDiff.gradient(logl, x), hesslog = x->ForwardDiff.hessian(logl, x), optimiser=Optim.LBFGS(), iterations=1000, show_every=-1)

#     laplace(logl, [x]; gradlogl=gradlogl, hesslog = hesslog, optimiser = optimiser, iterations = iterations, show_every = show_every)

# end

# function laplace(logl::Function, X::Array{Array{Float64,1},1}; gradlogl = x -> ForwardDiff.gradient(logl, x), hesslog = x->ForwardDiff.hessian(logl, x), optimiser=Optim.LBFGS(), iterations=1000, show_every=-1)

#     map(x->coreLaplace(logl, gradlogl, hesslog, x; iterations = iterations, optimiser = optimiser, show_every = show_every), X)

# end


# #-----------------------------------#
# # Call Mixed Variational Inference  #
# #-----------------------------------#

# function MVI(logl::Function, μ::Array{Float64,1}; gradlogl = x -> ForwardDiff.gradient(logl, x), optimiser=Optim.LBFGS(), laplaceiterations=10_000,  seed = 1, S = 100, iterations=1, numerical_verification = false, Stest=0, show_every=-1, inititerations=0)

#     MVI(logl, [μ]; gradlogl = gradlogl, seed = seed, S = S, optimiser=optimiser, laplaceiterations=laplaceiterations, iterations=iterations, numerical_verification = numerical_verification, Stest=Stest, show_every=show_every, inititerations=inititerations)

# end


# function MVI(logl::Function, μ::Array{Array{Float64,1},1}; gradlogl = x -> ForwardDiff.gradient(logl, x), optimiser=Optim.LBFGS(), laplaceiterations=10_000,  seed = 1, S = 100, iterations=1, numerical_verification = false, Stest=0, show_every=-1, inititerations=0)

#     LAposteriors = laplace(logl, μ; gradlogl = gradlogl, optimiser=optimiser, iterations=laplaceiterations, show_every=show_every)

#     coreMVI(logl, gradlogl, LAposteriors; seed = seed, S = S, optimiser = optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, inititerations=inititerations)

# end

# function MVI(logl::Function, LAposterior::MvNormal; gradlogl = x -> ForwardDiff.gradient(logl, x), optimiser=Optim.LBFGS(), laplaceiterations=10_000, seed = 1, S = 100, iterations=1, numerical_verification = false, Stest=0, show_every=-1, inititerations=0)

#     coreMVI(logl, gradlogl, [LAposterior]; seed = seed, S = S, optimiser = optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, inititerations=inititerations)

# end
