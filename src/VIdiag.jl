function coreVIdiag(logp::Function, μ₀::Vector, C₀diag::Vector; gradlogp = gradlogp, seed = seed, S = S, test_every = test_every, optimiser = optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every)

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

        local μ, Cdiag = unpack(param)

        local ℓ = elbo(μ, Cdiag, Ztrain)

        update!(trackELBO; newelbo = ℓ, μ = μ, C = Cdiag)

        return -1.0 * ℓ # Optim.optimise is minimising

    end


    function minauxiliary_grad(param)

        local μ, Cdiag = unpack(param)

        return -1.0 * elbo_grad(μ, Cdiag, Ztrain)  # Optim.optimise is minimising

    end


    # #----------------------------------------------------
    # # Functions for covariance and covariance root 
    # #----------------------------------------------------

    # getcov(Cdiag) = Diagonal(Cdiag.^2)
    
    # getcovroot(Cdiag) = Cdiag



    #----------------------------------------------------
    # Approximate evidence lower bound and its gradient
    #----------------------------------------------------

    function elbo(μ, Cdiag, Z)

        local aux = z -> logp(makeparameter(μ, Cdiag, z))

        Transducers.foldxt(+, Map(aux),  Z) / length(Z) + GaussianVariationalInference.entropy(Cdiag)

    end


    function partial_elbo_grad(μ, Cdiag, z)

        local g = gradlogp(makeparameter(μ, Cdiag, z))

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

    numerical_verification ? verifygradient(μ₀, C₀diag, elbo, minauxiliary_grad, unpack, Ztrain) : nothing
 

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
    # Call optimiser to minimise *negative* elbo
    #----------------------------------------------------
@show minauxiliary([μ₀; C₀diag])
    options = Optim.Options(extended_trace = false, store_trace = false, show_every = 1, show_trace = true,  iterations = iterations, g_tol = 1e-6)#, callback = trackELBO)

    result  = Optim.optimize(minauxiliary, gradhelper, [μ₀; C₀diag], optimiser, options)

    μopt, Copt = unpack(result.minimizer)


    #----------------------------------------------------
    # Return results
    #----------------------------------------------------

    return μopt, Copt, elbo(μopt, Copt, Ztrain)

end
