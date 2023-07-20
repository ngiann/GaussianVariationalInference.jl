function coreMVI(logp::Function, gradlogp::Function, μ₀; seed = 1, S = 100, optimiser = optimiser, iterations = 1, numerical_verification = false, Stest=0, show_every=-1, test_every = test_every, parallelmode = parallelmode)

    D = length(μ₀)


    #----------------------------------------------------
    # perform laplace approximation
    #----------------------------------------------------

    μLA, ΣLA = μ₀, get_covariance_at_mode(μ₀, x->-logp(x))

    VLA, EsqrtLA = eigendecomposition(ΣLA)


    #----------------------------------------------------
    # generate latent variables
    #----------------------------------------------------

    Ztrain = generatelatentZ(S = S, D = D, seed = seed)


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

        local ℓ = elbo(μ, Esqrt, Ztrain)
        
        update!(trackELBO; newelbo = ℓ, μ = μ, C = Esqrt)

        return -1.0 * ℓ

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

        local aux = VLA * Diagonal(Esqrt)

        local Σ = aux*aux'

        return (Σ + Σ') / 2

    end


    #----------------------------------------------------
    function getcovroot(Esqrt)
    #----------------------------------------------------

        VLA * Diagonal(Esqrt)

    end


    #----------------------------------------------------
    function elbo(μ, Esqrt, Z)
    #----------------------------------------------------

        local C = getcovroot(Esqrt)

        local aux = z -> logp(makeparameter(μ, C, z))

        evaluatesamples(aux, Z, Val(parallelmode)) + entropy(Esqrt)

    end

   
    #----------------------------------------------------
    function partial_elbo_grad(μ, C, z)
    #----------------------------------------------------

        local g = gradlogp(makeparameter(μ, C, z))

        [g;  VLA' * g .* z]

    end


    #----------------------------------------------------
    function elbo_grad(μ, Esqrt, Z)
    #----------------------------------------------------

        local C = getcovroot(Esqrt)

        
        # contribution of joint log-likelihood

        local aux = z -> partial_elbo_grad(μ, C, z)
        
        local gradμEsqrt = Transducers.foldxt(+, Map(aux), Z) / length(Z)


        # entropy contribution

        gradμEsqrt[D+1:2*D] += 1.0./Esqrt

        return gradμEsqrt

    end


    gradhelper(storage, param) = copyto!(storage, minauxiliary_grad(param))


    #----------------------------------------------------
    # Define callback function called at each iteration
    #----------------------------------------------------

    # We want to keep track of the best variational 
    # parameters encountered during the optimisation of
    # the elbo. Unfortunately, the otherwise superb
    # package Optim.jl does not provide a consistent way
    # accross different optimisers to do this.

    
    trackELBO = RecordELBOProgress(; μ = zeros(D), C = zeros(D), 
                                     Stest = Stest,
                                     show_every = show_every,
                                     test_every = test_every, 
                                     elbo = elbo, seed = seed)
    
     #----------------------------------------------------
    # Numerically verify gradient
    #----------------------------------------------------

    numerical_verification ? verifygradient(μ₀, EsqrtLA, elbo, minauxiliary_grad, unpack, Ztrain) : nothing
 


    #----------------------------------------------------
    # Call optimiser
    #----------------------------------------------------

    options = Optim.Options(extended_trace = false, store_trace = false, show_trace = false, show_every=show_every, iterations = iterations, g_tol = 1e-6, callback = trackELBO)

    result  = Optim.optimize(minauxiliary, gradhelper, [μLA ; EsqrtLA], optimiser, options)

    μopt, Esqrtopt = unpack(result.minimizer)


    #----------------------------------------------------
    # Return results
    #----------------------------------------------------

    Σopt = getcov(Esqrtopt)

    return MvNormal(μopt, Σopt), elbo(μopt, Esqrtopt, generatelatentZ(S = 10*S, D = D, seed = seed+2)), getcovroot(Esqrtopt)

end
