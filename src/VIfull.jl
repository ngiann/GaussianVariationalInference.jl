function coreVIfull(logl::Function, μarray::Array{Array{Float64,1},1}, Σarray::Array{Array{Float64, 2},1}; gradlogl = gradlogl, seed = 1, S = 100, optimiser=Optim.LBFGS(), iterations=1, numerical_verification = false, Stest=0, show_every=-1, inititerations=0)

    D = length(μarray[1])

    @assert(D == size(Σarray[1], 1) == size(Σarray[1], 2))

    @assert(length(μarray) == length(Σarray))

    @printf("Running VI with full covariance: seed=%d, S=%d, Stest=%d, D=%d for %d iterations\n", seed, S, Stest, D, iterations)


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

        return C*C'

    end


    #----------------------------------------------------
    function getcovroot(C)
    #----------------------------------------------------

        return C

    end


    #----------------------------------------------------
    function elbo(μ, C, Z)
    #----------------------------------------------------

        mean(map(z -> logl(μ .+ C*z), Z)) + ApproximateVI.entropy(C)

    end


    #----------------------------------------------------
    function elbo_grad(μ, C, Z)
    #----------------------------------------------------

        local Σroot = getcovroot(C)
        local gradC = (Σroot\I)' # entropy contribution
        local gradμ = zeros(eltype(μ), D)
        local S     = length(Z)

        for s=1:S
            g = gradlogl(μ .+ Σroot*Z[s])
            gradC += (1/S)*g*Z[s]'
            gradμ += g/S
        end

        return [vec(gradμ); vec(gradC)]

    end


    gradhelper(storage, param) = copyto!(storage, minauxiliary_grad(param))


    #----------------------------------------------------
    # Numerically verify gradient
    #----------------------------------------------------

    if numerical_verification

        local C = Matrix(cholesky(Σarray[1]).L)
        adgrad = ForwardDiff.gradient(minauxiliary, [μarray[1]; vec(C)])
        angrad = minauxiliary_grad([μarray[1];vec(C)])
        @printf("gradient from AD vs analytical gradient\n")
        display([vec(adgrad) vec(angrad)])
        @printf("maximum absolute difference is %f\n", maximum(abs.(vec(adgrad) - vec(angrad))))

    end


    #----------------------------------------------------
    # Evaluate initial solutions for few iterations
    #----------------------------------------------------

    initoptimise(μ, Σ) = Optim.optimize(minauxiliary, gradhelper, [μ; vec(Matrix(cholesky(Σ).L))], optimiser, Optim.Options(iterations = inititerations))

    results = if inititerations>0
        @showprogress "Initial search with random start " map(initoptimise, μarray, Σarray)
    else
        map(initoptimise, μarray, Σarray)
    end

    bestindex = argmin(map(r -> r.minimum, results))

    bestinitialsolution = results[bestindex].minimizer


    #----------------------------------------------------
    # Call optimiser
    #----------------------------------------------------

    options = Optim.Options(extended_trace = false, store_trace = false, show_trace = show_every > 0, show_every = show_every, iterations = iterations, g_tol = 1e-6)

    result  = Optim.optimize(minauxiliary, gradhelper, bestinitialsolution, optimiser, options)

    μopt, Copt = unpack(result.minimizer)


    #----------------------------------------------------
    # Return results
    #----------------------------------------------------

    Σopt = getcov(Copt)

    return MvNormal(μopt, Σopt), elbo(μopt, Copt, generatelatentZ(S = 10*S, D = D, seed = seed+2))

end
