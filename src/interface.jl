"""
# Basic use:

    q, logev, Croot = VI(logp, μ, σ=0.1; S = 100, iterations = 1, show_every = -1)

Returns approximate Gaussian posterior and log evidence.


# Arguments

A description of only the most basic arguments follows.

- `logp` is a function that expresses the (unnormalised) log-posterior, i.e. joint log-likelihood.
- `μ` is the initial mean of the approximating Gaussian posterior.
- `σ` specifies the initial covariance of the approximating Gaussian posterior as σ² * I . Default value is `√0.1`.
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
function VI(logp::Function, μ::Vector, Croot::Matrix; gradlogp = defaultgradient(μ), gradientmode = :gradientfree, seed::Int = 1, S::Int = 100, iterations::Int=1, numerical_verification::Bool = false, Stest::Int = 0, show_every::Int = -1, test_every::Int = -1)


    # check validity of arguments

    checkcommonarguments(seed, iterations, S, Stest, μ)

    @argcheck size(Croot, 1) == size(Croot, 2)                "Croot must be a square matrix"
    
    @argcheck length(μ)  == size(Croot, 1)  == size(Croot, 2) "dimensions of μ do not agree with dimensions of Croot"
    
    
    # pick optimiser and (re)define gradient of logp

    optimiser, gradlogp = pickoptimiser(μ, logp, gradlogp, gradientmode)


    # Call actual algorithm

    @printf("Running VI with full covariance: seed=%d, S=%d, Stest=%d, D=%d for %d iterations\n", seed, S, Stest, length(μ), iterations)
    
    reportnumberofthreads()

    coreVIfull(logp, μ, Croot; gradlogp = gradlogp, seed = seed, S = S, optimiser=optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, test_every = test_every)

end


function VI(logp::Function, μ::Vector, σ = sqrt(0.1); gradlogp = defaultgradient(μ), gradientmode = :gradientfree, seed::Int = 1, S::Int = 100, iterations::Int=1, numerical_verification::Bool = false, Stest::Int=0, show_every::Int = -1, test_every::Int = -1)

    @argcheck σ > 0    "σ must be ≥ 0"

    Croot = Matrix(σ*I, length(μ), length(μ)) # initial covariance

    VI(logp, μ, Croot; gradlogp = gradlogp, gradientmode = gradientmode, seed = seed, S = S, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, test_every = test_every)

end



#-----------------------------------------#
# Call mean field, two definitions follow #
#-----------------------------------------#

# Definition 1, general case: User specifies diagonal covariance root for defining initial covariance matrix

function VIdiag(logp::Function, μ::Vector, Cdiag::Vector; gradlogp = defaultgradient(μ), gradientmode = :gradientfree, seed::Int = 1, S::Int = 100, iterations::Int=1, numerical_verification::Bool = false, Stest::Int = 0, show_every::Int = -1, test_every::Int = -1, parallel::Bool = true, transform = identity)

    # check validity of arguments

    checkcommonarguments(seed, iterations, S, Stest, μ) 

    @argcheck length(Cdiag) == length(μ)  "Cdiag must be a vector the of same length as mean μ"   
    

    if transform !== identity
        
        local msg = @sprintf("A transform has been specified.\n")
        
        print(Crayon(foreground = :white, bold=true), msg, Crayon(reset = true))
        
        gradientmode == :gradientfree
        
    end


    # pick optimiser and (re)define gradient of logp

    optimiser, gradlogp = pickoptimiser(μ, logp, gradlogp, gradientmode)

    # Create out of the parallel argument a new argument of type symbol that is either :parallel or :serial
    # We use this internally to dispatch on value
    parallelmode = parallel ? :parallel : :serial


    # Call actual algorithm

    @printf("Running VI with diagonal covariance (mean field): seed=%d, S=%d, Stest=%d, D=%d for %d iterations\n", seed, S, Stest, length(μ), iterations)
    
    parallel ? reportnumberofthreads() : nothing

    coreVIdiag(logp, μ, Cdiag; gradlogp = gradlogp, seed = seed, S = S, test_every = test_every, optimiser = optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, parallelmode = parallelmode, transform = transform)

end

# Definition 2, special case: User specifies standard deviation fot defining initial diagonal covariance matrix.
#                             Calls definition 1.

function VIdiag(logp::Function, μ::Vector, σ = sqrt(0.1); gradlogp = defaultgradient(μ), gradientmode = :gradientfree, seed::Int = 1, S::Int = 100, iterations::Int=1, numerical_verification::Bool = false, Stest::Int = 0, show_every::Int = -1, test_every::Int = -1, parallel::Bool = true, transform = identity)

    @argcheck σ > 0  "σ must be ≥ 0"

    Cdiag = σ*ones(length(μ)) # initial diagonal covariance as vector

    VIdiag(logp, μ, Cdiag; gradlogp = gradlogp, gradientmode = gradientmode, seed = seed, S = S, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, test_every = test_every,  parallel =  parallel, transform = transform)

end



# #-----------------------------------#
# # Call Laplace                      #
# #-----------------------------------#

# function laplace(logp::Function, x::Array{Float64,1}; gradlogp = x -> ForwardDiff.gradient(logp, x), hesslog = x->ForwardDiff.hessian(logp, x), optimiser=Optim.LBFGS(), iterations=1000, show_every=-1)

#     laplace(logp, [x]; gradlogp=gradlogp, hesslog = hesslog, optimiser = optimiser, iterations = iterations, show_every = show_every)

# end

# function laplace(logp::Function, X::Array{Array{Float64,1},1}; gradlogp = x -> ForwardDiff.gradient(logp, x), hesslog = x->ForwardDiff.hessian(logp, x), optimiser=Optim.LBFGS(), iterations=1000, show_every=-1)

#     map(x->coreLaplace(logp, gradlogp, hesslog, x; iterations = iterations, optimiser = optimiser, show_every = show_every), X)

# end


# #-----------------------------------#
# # Call Mixed Variational Inference  #
# #-----------------------------------#

function MVI(logp::Function, μ::Vector; gradlogp = defaultgradient(μ), gradientmode = :gradientfree, seed::Int = 1, S::Int = 100, iterations::Int=1, numerical_verification = false, Stest=0, show_every=-1, test_every::Int = -1, parallel::Bool = true)

    
    # check validity of arguments
    
    checkcommonarguments(seed, iterations, S, Stest, μ) 
    
    # pick optimiser and (re)define gradient of logp
    
    optimiser, gradlogp = pickoptimiser(μ, logp, gradlogp, gradientmode)
    
    # Create out of the parallel argument a new argument of type symbol that is either :parallel or :serial
    # We use this internally to dispatch on value
    parallelmode = parallel ? :parallel : :serial
    
    # Call actual algorithm

    @printf("Running MVI with S=%d, D=%d for %d iterations\n", S, length(μ), iterations)

    parallel ? reportnumberofthreads() : nothing

    coreMVI(logp, gradlogp, μ; seed = seed, S = S, optimiser = optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, test_every = test_every, parallelmode = parallelmode)

end






# function VIrank1(logp::Function, μ::Vector, C::Matrix; gradlogp = defaultgradient(μ), gradientmode = :gradientfree, transform = identity, seed::Int = 1, seedtest::Int = 2, S::Int = 100, iterations::Int=1, numerical_verification::Bool = false, Stest::Int = 0, show_every::Int = -1, test_every::Int = -1)


#     # check validity of arguments

#     checkcommonarguments(seed, iterations, S, Stest, μ) 
       
#     @argcheck size(C, 1) == size(C, 2)                "C must be a square matrix"
    
#     @argcheck length(μ)  == size(C, 1)  == size(C, 2) "dimensions of μ do not agree with dimensions of C"
    

#     # pick optimiser and (re)define gradient of logp

#     optimiser, gradlogp = pickoptimiser(μ, logp, gradlogp, gradientmode)


#     # Call actual algorithm

#     @printf("Running VIrank1: seed=%d, S=%d, Stest=%d, D=%d for %d iterations\n", seed, S, Stest, length(μ), iterations)
    
#     reportnumberofthreads()

#     coreVIrank1(logp, μ, C; gradlogp = gradlogp, seed = seed, seedtest = seedtest+1000, S = S, test_every = test_every, optimiser = optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, transform = transform)

# end


function checkcommonarguments(seed, iterations, S, Stest, μ)

    # check validity of arguments

    @argcheck seed >= 0                 "seed must be ≥ 0"

    @argcheck iterations > 0            "iterations must be > 0"

    @argcheck S > 0                     "S must be > 0"
    
    @argcheck Stest >= 0                "Stest must be ≥ 0"
    
    @argcheck length(μ) >= 2            "VI works only for problems with two parameters or more"
   
end


function reportnumberofthreads()
    
    if Threads.nthreads() > 1
    
        @printf("\tRunning on %d available threads\n", Threads.nthreads())
    
    else

        @printf("\tRunning on single available thread. To use more threads start Julia with the option -t\n")

    end

end