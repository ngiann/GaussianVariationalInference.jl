function coreMVI(logl::Function, gradlogl::Function, LAposteriors; seed = 1, S = 100, optimiser = Optim.LBFGS(), iterations = 1, numerical_verification = false, Stest=0, show_every=-1, inititerations=0)

    D = length(mean(LAposteriors[1]))

    @assert(D == size(cov(LAposteriors[1]), 1) == size(cov(LAposteriors[1]), 2))



    @printf("Running MVI with S=%d, D=%d for %d iterations\n", S, D, iterations)


    #----------------------------------------------------
    # generate latent variables
    #----------------------------------------------------

    Ztrain = generatelatentZ(S = S, D = D, seed = seed)

    Ztest  = generatelatentZ(S = Stest, D = D, seed = seed+1)


    #----------------------------------------------------
    function unpack(param)
    #----------------------------------------------------

        @assert(length(param) == 2*D)

        local μ     = @view param[1:D]

        local Esqrt = @view param[D+1:2*D]

        return μ, Esqrt

    end


    #----------------------------------------------------
    function minauxiliary(V, param)
    #----------------------------------------------------

        local μ, Esqrt = unpack(param)

        return -1.0 * elbo(μ, Esqrt, V, Ztrain)

    end


    #----------------------------------------------------
    function minauxiliary_grad(V, param)
    #----------------------------------------------------

        local μ, Esqrt = unpack(param)

        return -1.0 * elbo_grad(μ, Esqrt, V, Ztrain)

    end


    #----------------------------------------------------
    function getcov(V, Esqrt)
    #----------------------------------------------------

        local aux = V * Diagonal(Esqrt)

        return aux*aux'

    end

    #----------------------------------------------------
    function getcovroot(V, Esqrt)
    #----------------------------------------------------

        return V * Diagonal(Esqrt)

    end


    #----------------------------------------------------
    function elbo(μ, Esqrt, V, Z)
    #----------------------------------------------------

        local C = getcovroot(V, Esqrt)

        mean(map(z -> logl(makeparameter(μ, C, z), Z)) + ApproximateVI.entropy(Esqrt)

    end


    #----------------------------------------------------
    function elbo_grad(μ, Esqrt, V, Z)
    #----------------------------------------------------

        local grad_μ     = zeros(D)

        local grad_Esqrt = zeros(D)

        local C          = getcovroot(V, Esqrt)

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

    gradhelper(V, storage, param) = copyto!(storage, minauxiliary_grad(V, param))


    #----------------------------------------------------
    # Numerically verify gradient
    #----------------------------------------------------

    if numerical_verification

        local V, Esqrt = eigendecomposition(cov(LAposteriors[1]))
        adgrad = ForwardDiff.gradient(x->minauxiliary(V, x), [mean(LAposteriors[1]); Esqrt])
        angrad = minauxiliary_grad(V, [mean(LAposteriors[1]); Esqrt])
        @printf("gradient from AD vs analytical gradient\n")
        display([vec(adgrad) vec(angrad)])
        @printf("maximum absolute difference is %f\n", maximum(abs.(vec(adgrad) - vec(angrad))))

    end


    #----------------------------------------------------
    # Evaluate initial solutions for few iterations
    #----------------------------------------------------

    function initoptimise(q)

        local V, Esqrt = eigendecomposition(cov(q))

        Optim.optimize(x->minauxiliary(V,x), (x1,x2)->gradhelper(V,x1,x2), [mean(q); Esqrt], optimiser, Optim.Options(iterations = inititerations))

    end

    results = if inititerations > 0
        @showprogress "Initial search with random start " map(initoptimise, LAposteriors)
    else
        map(initoptimise, LAposteriors)
    end

    bestindex = argmin(map(r -> r.minimum, results))

    bestinitialsolution = results[bestindex].minimizer

    V, Esqrt = eigendecomposition(cov(LAposteriors[bestindex]))


    #----------------------------------------------------
    # Call optimiser
    #----------------------------------------------------

    options = Optim.Options(extended_trace = false, store_trace = false, show_trace = true, show_every=show_every, iterations = iterations, g_tol = 1e-6)

    result  = Optim.optimize(x->minauxiliary(V, x), (x1,x2)->gradhelper(V,x1,x2), bestinitialsolution, optimiser, options)

    μopt, Esqrtopt = unpack(result.minimizer)


    #----------------------------------------------------
    # Return results
    #----------------------------------------------------

    Σopt = getcov(V, Esqrtopt)

    return MvNormal(μopt, Σopt), elbo(μopt, Esqrtopt, V, generatelatentZ(S = 10*S, D = D, seed = seed+2))

end
