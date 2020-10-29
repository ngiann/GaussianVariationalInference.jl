"""
    VI(logl, μ; S = 100, iterations = 1, show_every = -1)

    VI(logl, gradlogl, μ; S = 100, iterations = 1, show_every = -1)

    VI(logl, gradlogl, μ, Σ; S = 100, iterations = 1, show_every = -1)

Returns Gaussian posterior inferred via approximate VI and the log evidence

# Arguments

- logl is a function that expresses the joint log-likelihood
- gradlogl calculates the gradient for logl.
  If not given, it will be obtained via automatic differentiation.
- μ is the mean of the initial Gaussian posterior provided as an array.
  If instead, an array of μ is provided, they will all be tried out as initial
  candidate solutions and the best one, according to the lower bound, will be selected.
- Σ is the covariance of the initial Gaussian posterior.
  If unsecified, its  default values is 0.1*I.
- S is the number of drawn samples that approximate the lower bound integral
- Stest is the number of drawn samples to test the lower bound integral for overfitting
- show_every: report progress every "show_every" number of iterations

More arguments are explained in the README.md file.

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
function VI(logl::Function, μ; seed = 1, S = 100, optimiser=Optim.LBFGS(), iterations=1, numerical_verification = false,  Stest=0, show_every=NaN, inititerations=0)

    gradlogl(x)   = ForwardDiff.gradient(logl, x)

    gradlogl(v,x) = ForwardDiff.gradient!(v, logl, x)

    VI(logl, gradlogl, μ; seed = seed, S = S, optimiser = optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, inititerations=inititerations)

end



function VI(logl::Function, gradlogl::Function, μ::Array{Float64,1}; seed = 1, S = 100, optimiser=Optim.LBFGS(), iterations=1, numerical_verification = false, Stest=0, show_every=NaN, inititerations=0)


    VI(logl, gradlogl, μ, 0.1*Matrix(Diagonal(ones(length(μ)))); seed = seed, S = S, optimiser = optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, inititerations=inititerations)

end



function VI(logl::Function, gradlogl::Function, μ::Array{Array{Float64,1},1}; seed = 1, S = 100, optimiser=Optim.LBFGS(), iterations=1, numerical_verification = false, Stest=0, show_every=NaN, inititerations=0)


    VI(logl, gradlogl, μ, 0.1*Matrix(Diagonal(ones(length(μ[1])))); seed = seed, S = S, optimiser = optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, inititerations=inititerations)

end



function VI(logl::Function, gradlogl::Function, μ, Σ; seed = 1, S = 100, optimiser=Optim.LBFGS(), iterations=1, numerical_verification = false, Stest=0, show_every=NaN, inititerations=0)


    coreVI(logl, gradlogl, μ, Σ; seed = seed, S = S, optimiser = optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, inititerations=inititerations)

end



function coreVI(logl::Function, gradlogl::Function, μ::Array{Float64,1}, Σ::Array{Float64,2}; seed = 1, S = 100, optimiser=Optim.LBFGS(), iterations=1, numerical_verification = false, Stest=0, show_every=NaN, inititerations=0)


    coreVI(logl, gradlogl, [μ], Σ; seed = seed, S = S, optimiser=optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, inititerations=0)

end



function coreVI(logl::Function, gradlogl::Function, μarray::Array{Array{Float64,1},1}, Σ::Array{Float64,2}; seed = 1, S = 100, optimiser=Optim.LBFGS(), iterations=1, numerical_verification = false, Stest=0, show_every=NaN, inititerations=0)


    D = length(μarray[1])

    @assert(D == size(Σ, 1) == size(Σ, 2))

    @printf("Running VI with seed=%d, S=%d, Stest=%d, D=%d for %d iterations\n", seed, S, Stest, D, iterations)


    #----------------------------------------------------
    # Initialise matrix root
    #----------------------------------------------------

    C = Matrix(cholesky(Σ).L)


    #----------------------------------------------------
    # generate latent variables
    #----------------------------------------------------

    Ztrain = generatelatentZ(S = S, D = D, seed = seed)

    Ztest  = generatelatentZ(S = Stest, D = D, seed = seed+1)


    #----------------------------------------------------
    function unpack(param)
    #----------------------------------------------------

        @assert(length(param) == D+D*D)

        local μ = param[1:D]

        local C = reshape(param[D+1:D+D*D], D, D)

        return μ, C

    end


    #----------------------------------------------------
    function minauxiliary(param)
    #----------------------------------------------------

        local μ, C = unpack(param)

        return -1.0 * elbo(μ, C, Ztrain)

    end


    #----------------------------------------------------
    function minauxiliary_grad(param)
    #----------------------------------------------------

        local μ, C = unpack(param)

        return -1.0 * elbo_grad(μ, C, Ztrain)

    end


    #----------------------------------------------------
    function getcov(C)
    #----------------------------------------------------

        C*C'

    end


    #----------------------------------------------------
    function getcovroot(C)
    #----------------------------------------------------

        return C

    end


    #----------------------------------------------------
    function elbo(μ, C, Z)
    #----------------------------------------------------

        local Σ     = getcov(C)

        local Σroot = getcovroot(C)

        local l     = mean(map(z -> logl(μ .+ Σroot*z), Z))

        local H     = entropy(Σ)

        return l + H

    end


    #----------------------------------------------------
    function elbo_grad(μ, C, Z)
    #----------------------------------------------------

        local Σroot = getcovroot(C)
        local gradC = (Σroot\I)' # entropy contribution
        local gradμ = zeros(D)
        local S     = length(Z)

        for s=1:S
            g = gradlogl(μ .+ Σroot*Z[s])
            gradC += (1/S)*g*Z[s]'
            gradμ += g/S
        end

        return [vec(gradμ); vec(gradC)]

    end


    function gradhelper(storage, param)

        copyto!(storage, minauxiliary_grad(param))

    end


    #----------------------------------------------------
    # Numerically verify gradient
    #----------------------------------------------------

    if numerical_verification

        adgrad = ForwardDiff.gradient(minauxiliary, [μarray[1]; vec(C)])
        angrad = minauxiliary_grad([μarray[1];vec(C)])
        @printf("gradient from AD vs analytical gradient\n")
        display([vec(adgrad) vec(angrad)])
        @printf("maximum absolute difference is %f\n", maximum(abs.(vec(adgrad) - vec(angrad))))

    end


    #----------------------------------------------------
    # Monitor overfitting of latent variables Z
    #----------------------------------------------------

    history_elbo_Ztrain = zeros(0)

    history_elbo_Ztest  = zeros(0)

    history_iterations  = zeros(0)

    # history_parameters  = Array{ NamedTuple{(:μ, :C),Tuple{Array{Float64,1},Array{Float64,2}}}, 1}(undef, 0)

    function monitor(x)

        if mod(x.iteration, show_every) !== 0

            return false

        end

        local param = x.metadata["x"]

        local μ, C = unpack(param)

        # push!(history_parameters, (μ=μ,C=C))

        push!(history_elbo_Ztrain, elbo(μ, C, Ztrain))

        push!(history_elbo_Ztest,  Stest == 0 ? NaN : elbo(μ, C, Ztest))

        push!(history_iterations,  x.iteration)

        if Stest > 0
            @printf("\t iteration = %d, elbotest = %5.3f \t elbotrain = %5.3f\n", x.iteration, history_elbo_Ztest[end], history_elbo_Ztrain[end])
        else
            @printf("\t iteration = %d, elbotrain = %5.3f\n", x.iteration, history_elbo_Ztrain[end])
        end

        if Stest > 0
            display(  Plots.plot(history_iterations, history_elbo_Ztrain, label = "training", title="Monitoring elbo", legend = :topleft, lw=2) )
            display( Plots.plot!(history_iterations, history_elbo_Ztest,  label = "testing", lw=3) )
        end
        return false

    end


    #----------------------------------------------------
    # Evaluate initial solutions for few iterations
    #----------------------------------------------------

    initoptimise(x) = Optim.optimize(minauxiliary, gradhelper, [x; vec(C)], optimiser, Optim.Options(iterations = inititerations))

    results = if inititerations>0
        @showprogress "Initial search with random start " map(initoptimise, μarray)
    else
        map(initoptimise, μarray)
    end

    bestindex = argmin(map(r -> r.minimum, results))

    bestinitialsolution = results[bestindex].minimizer


    #----------------------------------------------------
    # Call optimiser
    #----------------------------------------------------

    # HACK
    copyoptim = Optim.Options(extended_trace = false, store_trace = false, show_trace = true, show_every=show_every, iterations = iterations)#, callback = monitor)

    result = Optim.optimize(minauxiliary, gradhelper, bestinitialsolution, optimiser, copyoptim)

    μopt, Copt = unpack(result.minimizer)


    #----------------------------------------------------
    # Return results
    #----------------------------------------------------

    negativelogevidence = result.minimum

    Σopt = getcov(Copt)

    return MvNormal(μopt, Σopt), elbo(μopt, Copt, generatelatentZ(S = 10*S, D = D, seed = seed+2))

end
