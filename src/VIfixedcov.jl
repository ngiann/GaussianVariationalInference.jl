function coreVIfixedcov(logl::Function, μ::Array{Float64,1}, fixedC::Matrix{Float64}; gradlogl = gradlogl, seed = 1, S = 100, optimiser=Optim.LBFGS(), iterations = 1, numerical_verification = false, Stest=0, show_every=-1, inititerations=0, adaptvariance=1)

    D = length(μ)

    @assert(D == size(fixedC, 1) == size(fixedC, 2))


    @printf("Running VI with fixed covariance with S=%d, D=%d for %d iterations\n", S, D, iterations)


    #----------------------------------------------------
    # generate latent variables
    #----------------------------------------------------

    Ztrain = generatelatentZ(S = S, D = D, seed = seed)

    Ztest  = generatelatentZ(S = Stest, D = D, seed = seed+1)


    #----------------------------------------------------
    function unpack(param)
    #----------------------------------------------------

        @assert(length(param) == D)

        local μ = param[1:D]

        return μ

    end


    #----------------------------------------------------
    function minauxiliary(param)
    #----------------------------------------------------

        local μ = unpack(param)

        return -1.0 * elbo(μ, Ztrain)

    end


    #----------------------------------------------------
    function minauxiliary_grad(param)
    #----------------------------------------------------

        local μ = unpack(param)

        return -1.0 * elbo_grad(μ, Ztrain)

    end


    #----------------------------------------------------
    function elbo(μ, Z)
    #----------------------------------------------------

        mean(map(z -> logl(μ .+ fixedC*z), Z)) + ApproximateVI.entropy(fixedC)

    end


    #----------------------------------------------------
    function elbo_grad(μ, Z)
    #----------------------------------------------------

        local gradμ = zeros(eltype(μ), D)

        local S     = length(Z)

        for s=1:S
            g      = gradlogl(μ .+ fixedC * Z[s])
            gradμ += g / S
        end

        return gradμ

    end


    gradhelper(storage, param) = copyto!(storage, minauxiliary_grad(param))


    #----------------------------------------------------
    # Numerically verify gradient
    #----------------------------------------------------

    if numerical_verification

        adgrad = ForwardDiff.gradient(minauxiliary, μ)
        angrad = minauxiliary_grad(μ)
        @printf("gradient from AD vs analytical gradient\n")
        display([vec(adgrad) vec(angrad)])
        @printf("maximum absolute difference is %f\n", maximum(abs.(vec(adgrad) - vec(angrad))))

    end


    #----------------------------------------------------
    # Call optimiser
    #----------------------------------------------------

    options = Optim.Options(extended_trace = false, store_trace = false, show_trace = true, show_every=show_every, iterations = iterations, g_tol = 1e-6)

    result  = Optim.optimize(minauxiliary, gradhelper, μ, optimiser, options)

    μopt = unpack(result.minimizer)


    #----------------------------------------------------
    # Return results
    #----------------------------------------------------

    return MvNormal(μopt, fixedC*fixedC'), elbo(μopt, generatelatentZ(S = 10*S, D = D, seed = seed+2))

end
