function coreVIdiag(logp::Function, μ₀::Vector, Cdiag::Vector; gradlogp = gradlogp, seed = seed, S = S, test_every = test_every, optimiser = optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, threshold = threshold)

    D = length(μ₀)

    #----------------------------------------------------
    # generate latent variables
    #----------------------------------------------------

    Ztrain = generatelatentZ(S = S, D = D, seed = seed)


    #----------------------------------------------------
    # Auxiliar function for handling parameters
    #----------------------------------------------------

    function unpack(param)
        
        @assert(length(param) == D+D)

        local μ = param[1:D]

        local Cdiag = reshape(param[D+1:D+D], D)

        return μ, Cdiag

    end
    

    #----------------------------------------------------
    # Objective and gradient functions for Optim.optimize
    #----------------------------------------------------

    function minauxiliary(param)

        local μ, C = unpack(param)

        local ℓ, = elbo(μ, C, Ztrain)

        return -1.0 * ℓ # Optim.optimise is minimising

    end


    function minauxiliary_grad(param)

        local μ, C = unpack(param)

        return -1.0 * elbo_grad(μ, C, Ztrain)  # Optim.optimise is minimising

    end


    #----------------------------------------------------
    # Functions for covariance and covariance root 
    #----------------------------------------------------
    
    getcov(Cdiag) = Diagonal(Cdiag.^2)
    
    getcovroot(Cdiag) = Cdiag


    #----------------------------------------------------
    # Approximate evidence lower bound and its gradient
    #----------------------------------------------------

    elbo(μ, C, Z) = GaussianVariationalInference.elbo(logp, μ, C, Z) # elbo function defined in elbo.jl

    function partial_elbo_grad(μ, Cdiag, z)

        local g = gradlogp(μ .+ Cdiag.*z)

        [g; vec(g.*z)]

    end


    function elbo_grad(μ, Cdiag, Z)
       
        local aux = z -> partial_elbo_grad(μ, Cdiag, z)
        
        local gradμCdiag = Transducers.foldxt(+, Map(aux), Z) / length(Z)

        # entropy contribution to covariance

        gradμCdiag[D+1:end] .+= vec(1.0 ./ Cdiag) 
        
        return gradμCdiag

    end


    # Package Optim requires that function for gradient has following signature

    gradhelper(storage, param) = copyto!(storage, minauxiliary_grad(param))


    #----------------------------------------------------
    # Numerically verify gradient
    #----------------------------------------------------

    numerical_verification ? verifygradient(μ₀, Cdiag, elbo, minauxiliary_grad, unpack, Ztrain) : nothing
 

    #----------------------------------------------------
    # Define callback function called at each iteration
    #----------------------------------------------------

    # We want to keep track of the best variational 
    # parameters encountered during the optimisation of
    # the elbo. Unfortunately, the otherwise superb
    # package Optim.jl does not provide a consistent way
    # accross different optimisers to do this.

    
    function testelbofunction(param)
        
        local μ, C = unpack(param)
        
        local f = z -> logp(makeparam(μ, C, z))
        
        local aux = map(f, [randn(D) for _ in 1:100])

        while sqrt(var(aux)/length(aux)) > threshold && length(aux) < 1_000_000

            auxmore = Transducers.tcollect(Map(f), [randn(D) for _ in 1:100])

            aux = vcat(aux, auxmore)

        end

        mean(aux) + entropy(C), sqrt(var(aux) / length(aux)), length(aux)

    end


    trackELBO = RecordELBOProgress(; initialparam = [μ₀; Cdiag], 
                                     show_every = show_every,
                                     test_every = test_every, 
                                     testelbofunction = testelbofunction, elbo = x -> elbo(unpack(x)..., Ztrain), unpack = unpack)
    
    


    #----------------------------------------------------
    # Call optimiser to minimise *negative* elbo
    #----------------------------------------------------

    options = Optim.Options(extended_trace = true, store_trace = false, show_every = 1, show_trace = false,  iterations = iterations, g_tol = 1e-4, callback = trackELBO)

    Optim.optimize(minauxiliary, gradhelper, [μ₀; Cdiag], optimiser, options)

    μopt, Copt = unpack(getbestsolution(trackELBO))


    #----------------------------------------------------
    # Return results
    #----------------------------------------------------

    return μopt, Copt, trackELBO

end
