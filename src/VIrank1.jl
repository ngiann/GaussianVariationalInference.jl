function coreVIrank1(logp::Function, μ₀::Vector, C₀::Matrix, u₀::Vector, v₀::Vector; gradlogp = gradlogp, seed = seed, S = S, test_every = test_every, optimiser = optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, transform = transform, seedtest = seedtest)

    D = length(μ₀)


    #----------------------------------------------------
    # generate latent variables
    #----------------------------------------------------

    Ztrain = generatelatentZ(S = S, D = D, seed = seed)


    #----------------------------------------------------
    # Auxiliar function for handling parameters
    #----------------------------------------------------

    function unpack(param)

        @assert(length(param) == 3D)

        local μ = param[1:D]

        local u, v = param[D+1:2D], param[2D+1:3D]

        return μ, u, v

    end


    #----------------------------------------------------
    # Objective and gradient functions for Optim.optimize
    #----------------------------------------------------

    function minauxiliary(param)

        local μ, u, v = unpack(param)

        local ℓ, = elbo(μ, u, v, Ztrain)

        return -1.0 * ℓ # Optim.optimise is minimising

    end


    function minauxiliary_grad(param)

        local μ, u, v = unpack(param)

        return -1.0 * elbo_grad(μ, u, v, Ztrain)  # Optim.optimise is minimising

    end


    #----------------------------------------------------
    # Functions for covariance and covariance root 
    #----------------------------------------------------

    function getcov(u, v)

        local aux = getcovroot(u, v)

        local Σ = aux*aux'

        (Σ + Σ') / 2 # for numerical stability

    end


    getcovroot(u, v) = C₀ + u*v'


    #----------------------------------------------------
    # Approximate evidence lower bound and its gradient
    #----------------------------------------------------

    elbo(μ, u, v, Z) = GaussianVariationalInference.elbo(logp, μ, getcovroot(u, v), Z) # elbo function defined in elbo.jl


    function partial_elbo_grad(μ, C, u, v, z)

        local g = gradlogp(μ .+ C*z)

        local gμ = g

        local gu = g*v'*z

        local gv = u'*g*z
            
        [gμ; vec(gu); vec(gv)]
        
    end


    function elbo_grad(μ, u, v, Z)

        local C = getcovroot(u, v)

        local aux = z -> partial_elbo_grad(μ, C, u, v, z)
        
        local gradμuv = Transducers.foldxt(+, Map(aux), Z) / length(Z)

        # entropy contribution to covariance

        gradμuv[D+1:2D] += C' \ v

        gradμuv[2D+1:3D] += C \ u
        
        return gradμuv

    end


    # Package Optim requires that function for gradient has following signature

    gradhelper(storage, param) = copyto!(storage, minauxiliary_grad(param))


    #----------------------------------------------------
    # Numerically verify gradient
    #----------------------------------------------------

    numerical_verification ? verifygradient(μ₀, u₀, v₀, elbo, minauxiliary_grad, unpack, Ztrain) : nothing

    
    #----------------------------------------------------
    # Define callback function called at each iteration
    #----------------------------------------------------

    # We want to keep track of the best variational 
    # parameters encountered during the optimisation of
    # the elbo. Unfortunately, the otherwise superb
    # package Optim.jl does not provide a consistent way
    # accross different optimisers to do this.

    
    function testelbofunction(param)
        
        local μ, u, v = unpack(param)

        local C = getcovroot(u, v)
        
        local f = z -> logp(makeparam(μ, C, z))
        
        local aux = map(f, [randn(D) for _ in 1:100])

        while sqrt(var(aux)/length(aux)) > 0.2

            auxmore = Transducers.tcollect(Map(f), [randn(D) for _ in 1:100])

            aux = vcat(aux, auxmore)

        end

        mean(aux) + entropy(C), sqrt(var(aux) / length(aux)), length(aux)

    end


    trackELBO = RecordELBOProgress(; initialparam = [μ₀; zeros(2D)], 
                                     show_every = show_every,
                                     test_every = test_every, 
                                     testelbofunction = testelbofunction, elbo = x -> elbo(unpack(x)..., Ztrain), unpack = unpack)
    


    #----------------------------------------------------
    # Call optimiser to minimise *negative* elbo
    #----------------------------------------------------

    options = Optim.Options(extended_trace = true, store_trace = false, show_trace = false,  iterations = iterations, g_tol = 1e-4, callback = trackELBO)

    Optim.optimize(minauxiliary, gradhelper, [μ₀; u₀; v₀], optimiser, options)

    μopt, uopt, vopt = unpack(getbestsolution(trackELBO)) # unpack(result.minimizer)


    #----------------------------------------------------
    # Return results
    #----------------------------------------------------

    return getbestelbo(trackELBO), μopt, uopt, vopt

end
