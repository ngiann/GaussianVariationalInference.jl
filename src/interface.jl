"""
# Basic use:

    q, logev = VI(logp, μ, σ²=0.1; S = 100, iterations = 1, show_every = -1)

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

- `q` is the approximating posterior returned as a ```Distributions.MvNormal``` type
- `logev` is the approximate log-evidence.


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
function VI(logp::Function, μ::AbstractVector, Σ::AbstractMatrix; gradlogp = defaultgradient(μ), gradientmode = :gradientfree, transform = identity, seed::Int = 1, S::Int = 100, iterations::Int=1, numerical_verification::Bool = false, Stest::Int = 0, show_every::Int = -1, test_every::Int = -1)


    # check validity of arguments

    @argcheck seed > 0           

    @argcheck iterations > 0    
    
    @argcheck S > 0         

    @argcheck size(Σ, 1) == size(Σ, 2)  "Σ must be a square matrix"
    
    @argcheck length(μ)  == size(Σ, 1)  == size(Σ, 2) "dimensions of μ do not agree with dimensions of Σ"
    
    @argcheck isposdef(Σ)               "Σ must be positive definite"
    
    @argcheck length(μ) >= 2            "VI works only for problems with two parameters and more"
    

    # check gradient arguments

    optimiser = NelderMead() # default optimiser


    if gradientmode == :forward
        
        gradlogp = x -> ForwardDiff.gradient(logp, x)

        optimiser = LBFGS() # optimiser to be used with gradient calculated wiht automatic differentiation

    elseif gradientmode == :provided

        if any(isnan.(gradlogp(μ)))
            
            error("provided gradient returns NaN when evaluate at provided μ")

        end


        # check user provided gradient against gradient calculated via automatic differention via ForwardDiff.jl

        if numerical_verification

            local user_grad, autodiff_grad = gradlogp(μ), ForwardDiff.gradient(logp, μ)
        
            display([user_grad autodiff_grad])

            @printf("Left column above is the user provided gradient evaluated at μ\n")
            
            @printf("Right column above is user AD gradient evaluated at μ\n")
            
            local discr = maximum(abs.(user_grad .- autodiff_grad))

            local msg = @sprintf("Greatest elementwise difference in absolute magnitude is %.8f\n\n", discr)
            
            local clr = discr > 1e-4 ? :red : :cyan

            print(Crayon(foreground = clr, bold=true), msg, Crayon(reset = true))
            
        end

        optimiser = LBFGS() # optimiser to be used with user provided gradient

    elseif gradientmode == :gradientfree
        
        optimiser = NelderMead() # optimiser when no gradient provided

    else

        error("invalid specification of argument gradientmode")

    end


    if transform !== identity

        local msg = @sprintf("A transform has been specified. Only supported optimisation for now is :gradientfree for now.\n")
        
        print(Crayon(foreground = :yellow, bold=true), msg, Crayon(reset = true))

        gradientmode == :gradientfree

    end
    

    # Call actual algorithm

    @printf("Running VI with full covariance: seed=%d, S=%d, Stest=%d, D=%d for %d iterations\n", seed, S, Stest, length(μ), iterations)

    coreVIfull(logp, μ, Σ; gradlogp = gradlogp, seed = seed, S = S, optimiser=optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, test_every = test_every, transform = transform)

end


function VI(logp::Function, μ::AbstractVector, σ² = 0.1; gradlogp = defaultgradient(μ), gradientmode = :gradientfree, transform = identity, seed::Int = 1, S::Int = 100, iterations::Int=1, numerical_verification::Bool = false, Stest::Int=0, show_every::Int = -1, test_every::Int = -1)

    @argcheck σ² > 0
    
    # initial covariance

    D = length(μ)
    
    Σ = Matrix(σ²*I, length(μ), length(μ))

    VI(logp, μ, Σ; gradlogp = gradlogp, gradientmode = gradientmode, transform = transform, seed = seed, S = S, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, test_every = test_every)

end


function VI(logp::Function, initgaussian::AbstractMvNormal; gradlogp = defaultgradient(mean(initgaussian)), gradientmode = :gradientfree, seed::Int = 1, S::Int = 100, iterations::Int = 1, numerical_verification::Bool = false, Stest::Int = 0, show_every::Int = -1,  test_every::Int = -1)

    VI(logp, mean(initgaussian), cov(initgaussian); gradlogp = gradlogp, gradientmode = gradientmode, seed = seed, S = S, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, test_every = test_every)

end




#-----------------------------------#
# Call mean field                   #
#-----------------------------------#

function VIdiag(logp::Function, μ::Vector, Cdiag::Vector = 0.1*ones(length(μ)); gradlogp = defaultgradient(μ), gradientmode = :gradientfree, seed::Int = 1, S::Int = 100, iterations::Int=1, numerical_verification::Bool = false, Stest::Int = 0, show_every::Int = -1, test_every::Int = -1, transform = identity)

    # check validity of arguments

    # checkcommonarguments(seed, iterations, S, Stest, μ) 

    @argcheck length(Cdiag) == length(μ)  "Cdiag must be a vector the of same length as mean μ"   

    # pick optimiser and (re)define gradient of logp

    optimiser = NelderMead()

    if transform !== identity

        local msg = @sprintf("A transform has been specified. Only supported optimisation for now is :gradientfree for now.\n")
        
        print(Crayon(foreground = :yellow, bold=true), msg, Crayon(reset = true))

        gradientmode == :gradientfree

    end

    # Call actual algorithm

    @printf("Running VI with diagonal covariance (mean field): seed=%d, S=%d, Stest=%d, D=%d for %d iterations\n", seed, S, Stest, length(μ), iterations)
    # reportnumberofthreads()

    coreVIdiag(logp, μ, Cdiag; gradlogp = gradlogp, seed = seed, S = S, test_every = test_every, optimiser = optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, transform = transform)

end


function VIdiag(logp::Function, μ::Vector, σ::Float64 = 0.1; gradlogp = defaultgradient(μ), gradientmode = :gradientfree, seed::Int = 1, S::Int = 100, iterations::Int=1, numerical_verification::Bool = false, Stest::Int = 0, show_every::Int = -1, test_every::Int = -1, transform = identity)

    @argcheck σ > 0  "σ must be ≥ 0"

    Cdiag = σ*ones(length(μ)) # initial diagonal covariance as vector

    VIdiag(logp, μ, Cdiag; gradlogp = gradlogp, gradientmode = gradientmode, seed = seed, S = S, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, test_every = test_every, transform = transform)

end



# #-----------------------------------#
# # Call VI with spherical covariance #
# #-----------------------------------#

# function VIfixedcov(logp::Function, μ::Array{Float64,1}, fixedC::Array{Float64,2}; gradlogp = x -> ForwardDiff.gradient(logp, x), optimiser=Optim.LBFGS(), seed = 1, S = 100, iterations=1, numerical_verification = false, Stest=0, show_every=-1, inititerations=0, adaptvariance = 1)

#     coreVIfixedcov(logp, μ, fixedC, gradlogp = gradlogp, seed = seed, S = S, optimiser = optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, inititerations = inititerations, adaptvariance = adaptvariance)

# end


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
