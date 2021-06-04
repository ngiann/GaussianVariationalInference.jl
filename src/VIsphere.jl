function coreVIsphere(logl::Function, μarray::Array{Array{Float64,1},1}, σarray::Array{Float64,1}; gradlogl = gradlogl, seed = 1, S = 100, optimiser=Optim.LBFGS(), iterations = 1, numerical_verification = false, Stest=0, show_every=-1, inititerations=0, adaptvariance=1)

    D = length(μarray[1])

    @assert(length(μarray) == length(σarray))

    @printf("Running VI spherical with S=%d, D=%d for %d iterations\n", S, D, iterations)


    #----------------------------------------------------
    # generate latent variables
    #----------------------------------------------------

    Ztrain = generatelatentZ(S = S, D = D, seed = seed)

    Ztest  = generatelatentZ(S = Stest, D = D, seed = seed+1)


    #----------------------------------------------------
    function unpack(param)
    #----------------------------------------------------

        @assert(length(param) == D+1)

        local μ = param[1:D]

        local σ = param[1+D] # ❗ Note: we don't constrain the sign ❗

        return μ, σ

    end


    #----------------------------------------------------
    function minauxiliary(param)
    #----------------------------------------------------

        local μ, σ = unpack(param)

        return -1.0 * elbo(μ, σ, Ztrain)

    end


    #----------------------------------------------------
    function minauxiliary_grad(param)
    #----------------------------------------------------

        local μ, σ = unpack(param)

        return -1.0 * elbo_grad(μ, σ, Ztrain)

    end


    #----------------------------------------------------
    function elbo(μ, σ, Z)
    #----------------------------------------------------

        mean(map(z -> logl(μ .+ σ.*z), Z)) + ApproximateVI.entropy(σ*ones(D))

    end


    #----------------------------------------------------
    function elbo_grad(μ, σ, Z)
    #----------------------------------------------------

        local gradlogσ = D / σ # entropy contribution

        local gradμ = zeros(eltype(μ), D)

        local S     = length(Z)

        for s=1:S
            g         = gradlogl(μ .+ σ .* Z[s])
            gradlogσ += sum(g .* Z[s] / S)
            gradμ    += g / S
        end

        return [vec(gradμ); gradlogσ*adaptvariance]

    end


    gradhelper(storage, param) = copyto!(storage, minauxiliary_grad(param))


    #----------------------------------------------------
    # Numerically verify gradient
    #----------------------------------------------------

    if numerical_verification

        local σ = σarray[1]
        adgrad = ForwardDiff.gradient(minauxiliary, [μarray[1]; σ])
        angrad = minauxiliary_grad([μarray[1]; σ])
        @printf("gradient from AD vs analytical gradient\n")
        display([vec(adgrad) vec(angrad)])
        @printf("maximum absolute difference is %f\n", maximum(abs.(vec(adgrad) - vec(angrad))))

    end


    #----------------------------------------------------
    # Evaluate initial solutions for few iterations
    #----------------------------------------------------

    initoptimise(μ, σ) = Optim.optimize(minauxiliary, gradhelper, [μ; σ], optimiser, Optim.Options(iterations = inititerations))

    results = if inititerations > 0

        @showprogress "Initial search with random start " map(initoptimise, μarray, σarray)

    else

        map(initoptimise, μarray, σarray)

    end

    bestindex = argmin(map(r -> r.minimum, results))

    bestinitialsolution = results[bestindex].minimizer


    #----------------------------------------------------
    # Call optimiser
    #----------------------------------------------------

    options = Optim.Options(extended_trace = false, store_trace = false, show_trace = true, show_every=show_every, iterations = iterations, g_tol = 1e-6)

    result  = Optim.optimize(minauxiliary, gradhelper, bestinitialsolution, optimiser, options)

    μopt, σopt = unpack(result.minimizer)


    #----------------------------------------------------
    # Return results
    #----------------------------------------------------

    return MvNormal(μopt, σopt), elbo(μopt, σopt, generatelatentZ(S = 10*S, D = D, seed = seed+2))

end
