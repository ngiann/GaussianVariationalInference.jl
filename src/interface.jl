"""
# Basic use:

    q, logev, Croot = VI(logp, μ, σ²=0.1; S = 100, iterations = 1, show_every = -1)

Returns approximate Gaussian posterior and log evidence.


# Arguments

A description of only the most basic arguments follows.

- `logp` is a function that expresses the (unnormalised) log-posterior, i.e. joint log-likelihood.
- `μ` is the initial mean of the approximating Gaussian posterior.
- `σ²` specifies the initial covariance of the approximating Gaussian posterior as σ² * I . Default value is `0.1`.
- `S` is the number of drawn samples that approximate the lower bound integral.
- `iterations` specifies for how many iterations to run optimisation on the lower bound (elbo).
- `show_every`: report progress every `show_every` number of iterations. If set to value smaller than `1`, then no progress will be reported.

# Outputs

- `q` is the approximating posterior returned as a ```Distributions.MvNormal``` type.
- `logev` is the approximate log-evidence.
- `Croot` is the matrix root of the posterior covariance.


# Example

```julia-repl
# infer posterior of Bayesian linear regression, compare to exact result
julia> using LinearAlgebra, Distributions
julia> D = 4; X = randn(D, 1000); W = randn(D); β = 0.3; α = 1.0;
julia> Y = vec(W'*X); Y += randn(size(Y))/sqrt(β);
julia> Sn = inv(α*I + β*(X*X')) ; mn = β*Sn*X*Y; # exact posterior
julia> posterior, logev, = VI( w -> logpdf(MvNormal(vec(w'*X), sqrt(1/β)), Y) + logpdf(MvNormal(zeros(D),sqrt(1/α)), w), randn(D); S = 1_000, iterations = 15);
julia> display([mean(posterior) mn])
julia> display([cov(posterior)  Sn])
julia> display(logev) # display negative log evidence
```

"""
function VI(logp::Function, μ::Vector, Croot::Matrix; gradlogp = defaultgradient(μ), gradientmode = :gradientfree, seed::Int = 1, S::Int = 100, iterations::Int=1, numerical_verification::Bool = false, Stest::Int = 0, show_every::Int = -1, test_every::Int = -1, threshold::Float64 = 0.2)


    # check validity of arguments

    checkcommonarguments(seed, iterations, S, Stest, μ, threshold)

    @argcheck size(Croot, 1) == size(Croot, 2)                "Croot must be a square matrix"
    
    @argcheck length(μ)  == size(Croot, 1)  == size(Croot, 2) "dimensions of μ do not agree with dimensions of Croot"
    

    # pick optimiser and (re)define gradient of logp

    optimiser, gradlogp = pickoptimiser(μ, logp, gradlogp, gradientmode)


    # Call actual algorithm

    print(Crayon(foreground = :white, bold=true), @sprintf("Running VI with full covariance: seed=%d, S=%d, Stest=%d, D=%d for %d iterations\n", seed, S, Stest, length(μ), iterations), Crayon(reset = true))
    
    reportnumberofthreads()

    coreVIfull(logp, μ, Croot; gradlogp = gradlogp, seed = seed, S = S, optimiser=optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, test_every = test_every, threshold = threshold)

end


function VI(logp::Function, μ::Vector, σ = 0.1; gradlogp = defaultgradient(μ), gradientmode = :gradientfree, seed::Int = 1, S::Int = 100, iterations::Int=1, numerical_verification::Bool = false, Stest::Int=0, show_every::Int = -1, test_every::Int = -1, threshold::Float64 = 0.2)

    @argcheck σ > 0    "σ must be ≥ 0"

    Croot = Matrix(σ*I, length(μ), length(μ)) # initial covariance

    VI(logp, μ, Croot; gradlogp = gradlogp, gradientmode = gradientmode, seed = seed, S = S, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, test_every = test_every, threshold = threshold)

end


#-----------------------------------#
#         Call mean field           #
#-----------------------------------#

function VIdiag(logp::Function, μ::Vector, Cdiag::Vector = 0.1*ones(length(μ)); gradlogp = defaultgradient(μ), gradientmode = :gradientfree, seed::Int = 1, S::Int = 100, iterations::Int=1, numerical_verification::Bool = false, Stest::Int = 0, show_every::Int = -1, test_every::Int = -1, threshold::Float64 = 0.2)

    # check validity of arguments

    checkcommonarguments(seed, iterations, S, Stest, μ, threshold) 

    @argcheck length(Cdiag) == length(μ)         "Cdiag must be a vector the of same length as mean μ"
    
    @argcheck isposdef(Diagonal(Cdiag.*Cdiag))   "Cdiag must be positive definite"
    

    # pick optimiser and (re)define gradient of logp

    optimiser, gradlogp = pickoptimiser(μ, logp, gradlogp, gradientmode)


    # Call actual algorithm

    print(Crayon(foreground = :white, bold=true), @sprintf("Running VI with diagonal covariance (mean field): seed=%d, S=%d, Stest=%d, D=%d for %d iterations\n", seed, S, Stest, length(μ), iterations), Crayon(reset = true))
    
    reportnumberofthreads()

    coreVIdiag(logp, μ, Cdiag; gradlogp = gradlogp, seed = seed, S = S, test_every = test_every, optimiser = optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, threshold = threshold)

end


function VIdiag(logp::Function, μ::Vector, σ²::Float64 = 0.1; gradlogp = defaultgradient(μ), gradientmode = :gradientfree, seed::Int = 1, S::Int = 100, iterations::Int=1, numerical_verification::Bool = false, Stest::Int = 0, show_every::Int = -1, test_every::Int = -1, threshold::Float64 = 0.2)

    @argcheck σ² > 0  "σ² must be ≥ 0"

    Σdiag = σ²*ones(length(μ)) # initial diagonal covariance as vector

    VIdiag(logp, μ, Σdiag; gradlogp = gradlogp, gradientmode = gradientmode, seed = seed, S = S, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, test_every = test_every, threshold = threshold)

end





# #-----------------------------------#
# # Call Mixed Variational Inference  #
# #-----------------------------------#

# function MVI(logp::Function, μ::Array{Float64,1}; gradlogp = x -> ForwardDiff.gradient(logp, x), optimiser=Optim.LBFGS(), laplaceiterations=10_000,  seed = 1, S = 100, iterations=1, numerical_verification = false, Stest=0, show_every=-1, inititerations=0)

#     MVI(logp, [μ]; gradlogp = gradlogp, seed = seed, S = S, optimiser=optimiser, laplaceiterations=laplaceiterations, iterations=iterations, numerical_verification = numerical_verification, Stest=Stest, show_every=show_every, inititerations=inititerations)

# end


# function MVI(logp::Function, μ::Array{Array{Float64,1},1}; gradlogp = x -> ForwardDiff.gradient(logp, x), optimiser=Optim.LBFGS(), laplaceiterations=10_000,  seed = 1, S = 100, iterations=1, numerical_verification = false, Stest=0, show_every=-1, inititerations=0)

#     LAposteriors = laplace(logp, μ; gradlogp = gradlogp, optimiser=optimiser, iterations=laplaceiterations, show_every=show_every)

#     coreMVI(logp, gradlogp, LAposteriors; seed = seed, S = S, optimiser = optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, inititerations=inititerations)

# end

# function MVI(logp::Function, LAposterior::MvNormal; gradlogp = x -> ForwardDiff.gradient(logp, x), optimiser=Optim.LBFGS(), laplaceiterations=10_000, seed = 1, S = 100, iterations=1, numerical_verification = false, Stest=0, show_every=-1, inititerations=0)

#     coreMVI(logp, gradlogp, [LAposterior]; seed = seed, S = S, optimiser = optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, inititerations=inititerations)

# end



#-----------------------------------#
#          Call low rank            #
#-----------------------------------#

function VIrank1(logp::Function, μ::Vector, C::Matrix; gradlogp = defaultgradient(μ), gradientmode = :gradientfree, transform = identity, seed::Int = 1, seedtest::Int = 2, S::Int = 100, iterations::Int=1, numerical_verification::Bool = false, Stest::Int = 0, show_every::Int = -1, test_every::Int = -1, threshold::Float64 = 0.2)

    D = length(μ)

    rg = MersenneTwister(seed)

    u = 0.1 * randn(rg, D)

    v = 0.1 * randn(rg, D)

    VIrank1(logp, μ, C, u, v; gradlogp = gradlogp, gradientmode = gradientmode, transform = transform, seed = seed, seedtest = seedtest, S = S, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, test_every = test_every, threshold = threshold)

end


function VIrank1(logp::Function, μ::Vector, C::Matrix, u::Vector, v::Vector; gradlogp = defaultgradient(μ), gradientmode = :gradientfree, transform = identity, seed::Int = 1, seedtest::Int = 2, S::Int = 100, iterations::Int=1, numerical_verification::Bool = false, Stest::Int = 0, show_every::Int = -1, test_every::Int = -1, threshold = threshold)

    # check validity of arguments

    checkcommonarguments(seed, iterations, S, Stest, μ, threshold) 
       
    @argcheck size(C, 1) == size(C, 2)                "C must be a square matrix"
    
    @argcheck length(μ)  == size(C, 1)  == size(C, 2) "dimensions of μ and C do not agree"
    
    @argcheck length(μ)  == length(v)                 "μ and v must agree in dimensions"

    @argcheck length(μ)  == length(u)                 "μ and u must agree in dimensions"

    # pick optimiser and (re)define gradient of logp

    optimiser, gradlogp = pickoptimiser(μ, logp, gradlogp, gradientmode)


    # Call actual algorithm

    print(Crayon(foreground = :white, bold=true), @sprintf("Running VIrank1: seed=%d, S=%d, Stest=%d, D=%d for %d iterations\n", seed, S, Stest, length(μ), iterations), Crayon(reset = true))
    
    reportnumberofthreads()

    coreVIrank1(logp, μ, C, u, v; gradlogp = gradlogp, seed = seed, seedtest = seedtest+1000, S = S, test_every = test_every, optimiser = optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, transform = transform, threshold = threshold)

end


function checkcommonarguments(seed, iterations, S, Stest, μ, threshold)

    # check validity of arguments

    @argcheck seed >= 0                 "seed must be ≥ 0"

    @argcheck iterations > 0            "iterations must be > 0"

    @argcheck S > 0                     "S must be > 0"
    
    @argcheck Stest >= 0                "Stest must be ≥ 0"
    
    @argcheck length(μ) >= 2            "VI works only for problems with two parameters and more"

    @argcheck threshold >= 0            "threshold must be positive. Minimum recommended value is 0.1"
   
end


function reportnumberofthreads()
    
    if Threads.nthreads() > 1
        
        print(Crayon(foreground = :light_blue, bold=false), @sprintf("\t Number of available threads is %d\n", Threads.nthreads()), Crayon(reset = true))
    
    else
    
        print(Crayon(foreground = :light_blue, bold=false), @sprintf("\t Single thread available. To use multiple threads start julia with flag -t <number of threads>\n"), Crayon(reset = true))    
    
    end

end