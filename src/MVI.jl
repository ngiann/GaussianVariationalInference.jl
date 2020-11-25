"""
    MVI(logl; S = 100, D = D, maxiter = 1)

    MVI(logl, gradlogl; S = 100, D = D, maxiter = 1)

    MVI(logl, gradlogl, μ; S = 100, D = D, maxiter = 1)

    MVI(logl, gradlogl, μ, Σ; S = 100, D = D, maxiter = 1)

Returns mean and covariance of approximate Gaussian posterior inferred via MVI

# Arguments

- logl is a function that expresses the joint log-likelihood
- gradlogl calculates the gradient for logl. If not given, it will be obtained via automatic differentiation.
- S is the number of drawn samples that approximate the lower bound integral
- D is the dimension of the parameter


# Examples
```julia-repl
# Trivial example: approximate Gaussian with Gaussian
julia> MVI(x->logpdf(MvNormal(zeros(2), [1.0 0.3;0.3 1.0]),x); S = 200, D = 2, maxiter = 50)
([0.0848098504274756, 0.08305942802203821], [0.8922994469183307 0.27631010716184484; 0.27631010716184484 0.9603766726205719])
```
"""
function MVI(logl::Function; S = 100, D = D, maxiter = 1, optimiser = Optim.LBFGS())

    gradlogl(x)   = ForwardDiff.gradient(logl, x)

    gradlogl(v,x) = ForwardDiff.gradient!(v, logl, x)

    MVI(logl, gradlogl; S = S, D = D, maxiter = maxiter, optimiser = optimiser)

end


##############################################################
function MVI(logl::Function, gradlogl::Function; S = 100, D = D, maxiter = 1, optimiser = Optim.LBFGS())
##############################################################

    μLA, ΣLA = laplace(randn(D), logl, gradlogl)

    MVI(logl, gradlogl, μLA, ΣLA; S = S, D = D, maxiter = maxiter, optimiser = optimiser)

end


##############################################################
function MVI(logl::Function, gradlogl::Function, μ::Array{Float64,1}; S = 100, D = D, maxiter = 1, optimiser = Optim.LBFGS())
##############################################################

    μLA, ΣLA = laplace(μ, logl, gradlogl)

    MVI(logl, gradlogl, μLA, ΣLA; S = S, D = D, maxiter = maxiter, optimiser = optimiser)

end


##############################################################
function MVI(logl::Function, μ::Array{Float64,1}; S = 100, D = D, maxiter = 1, optimiser = Optim.LBFGS())
##############################################################

    gradlogl(x)   = ForwardDiff.gradient(logl, x)

    gradlogl(v,x) = ForwardDiff.gradient!(v, logl, x)

    μLA, ΣLA = laplace(μ, logl, gradlogl)

    MVI(logl, gradlogl, μLA, ΣLA; S = S, D = D, maxiter = maxiter, optimiser = optimiser)

end


##################################################################
function MVI(logl::Function, gradlogl::Function, μ, Σ; S = 100, D = D, maxiter = 1, optimiser = Optim.LBFGS())
##################################################################

    coreMVI(logl, gradlogl, μ, Σ; S = S, D = D, maxiter = maxiter, optimiser = optimiser)

end


##################################################################
function coreMVI(logl::Function, gradlogl::Function, μ, Σ; S = 100, D = D, maxiter = 1, optimiser = Optim.LBFGS())
##################################################################

    @printf("Running MVI with S=%d, D=%d for %d iterations\n", S, D, maxiter)

    V, Esqrt = eigendecomposition(Σ)

    #----------------------------------------------------
    # generate latent variables
    #----------------------------------------------------

    Ztrain = generatelatentZ(S = S, D = D)


    #----------------------------------------------------
    function unpack(param)
    #----------------------------------------------------

        @assert(length(param) == 2*D)

        local μ     = @view param[1:D]

        local Esqrt = @view param[D+1:2*D]

        return μ, Esqrt

    end


    #----------------------------------------------------
    function minauxiliary(param)
    #----------------------------------------------------

        local μ, Esqrt = unpack(param)

        return -1.0 * elbo(μ, Esqrt, Ztrain)

    end


    #----------------------------------------------------
    function minauxiliary_grad(param)
    #----------------------------------------------------

        local μ, Esqrt = unpack(param)

        return -1.0 * elbo_grad(μ, Esqrt, Ztrain)

    end


    #----------------------------------------------------
    function getcov(Esqrt)
    #----------------------------------------------------

        return V * Diagonal(Esqrt.^2) * V'

    end

    #----------------------------------------------------
    function getcovroot(Esqrt)
    #----------------------------------------------------

        return V * Diagonal(Esqrt)

    end


    #----------------------------------------------------
    function elbo(μ, Esqrt, Z)
    #----------------------------------------------------

        local C = getcovroot(Esqrt)

        local l = mean(map(z -> logl(μ .+ C*z), Z))

        local H = entropy_sqrt_eigenvalues(Esqrt)


        return l + H

    end


    #----------------------------------------------------
    function elbo_grad(μ, Esqrt, Z)
    #----------------------------------------------------

        local grad_μ     = zeros(D)

        local grad_Esqrt = zeros(D)

        local C          = getcovroot(Esqrt)

        local S          = length(Z)

        for i=1:S

            local aux = gradlogl(μ .+ C * Z[i])

            grad_μ += aux / S

            grad_Esqrt .+= V' * aux .* Z[i] / S

        end

        # logdet contribution in entropy

        grad_Esqrt .+= 1.0./Esqrt

        return [grad_μ; grad_Esqrt]

    end

    # Numerically verify gradient
    # param = randn(2*D)
    # @show minauxiliary_grad(param)
    # @show ForwardDiff.gradient(minauxiliary, param)

    opt      = Optim.Options(show_trace=true,  show_every=1, iterations=maxiter)

    result   = Optim.optimize(minauxiliary, [μ; Esqrt], optimiser, opt)

    μopt, Esqrtopt = unpack(result.minimizer)

    return MvNormal(μopt, getcov(Esqrtopt)), elbo(μopt, Esqrtopt, generatelatentZ(S = 10*S, D = D, seed = seed+2))

end
