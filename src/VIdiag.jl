function coreVIdiag(logp::Function, μ₀::AbstractArray{T, 1}, Cdiag::AbstractArray{T, 1}; gradlogp = gradlogp, seed = seed, S = S, test_every = test_every, optimiser = optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every) where T

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

        local ℓ, stdℓ = elbo(μ, Cdiag, Ztrain)

        update!(trackELBO; newelbo = ℓ, newelbo_std = stdℓ, μ = μ, C = Cdiag)

        return -1.0 * ℓ # Optim.optimise is minimising

    end


    function minauxiliary_grad(param)

        local μ, Cdiag = unpack(param)

        return -1.0 * elbo_grad(μ, Cdiag, Ztrain)  # Optim.optimise is minimising

    end


    #----------------------------------------------------
    # Functions for covariance and covariance root 
    #----------------------------------------------------

    
    function getcov(Cdiag)
        
        Diagonal(Cdiag.^2)
    
    end


    function getcovroot(Cdiag)
    
        return Cdiag

    end


    #----------------------------------------------------
    # Approximate evidence lower bound and its gradient
    #----------------------------------------------------

    function elbo(μ, Cdiag, Z)

        local aux = map(z -> logp(makeparam(μ, Cdiag, z)), Z)

        mean(aux) + GaussianVariationalInference.entropy(Cdiag), sqrt(var(aux)/length(Z))

    end


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

    
    trackELBO = RecordELBOProgress(; μ = zeros(D), C = zeros(D), 
                                     Stest = Stest,
                                     show_every = show_every,
                                     test_every = test_every, 
                                     logp = logp, seed = seed)
    


    #----------------------------------------------------
    # Call optimiser to minimise *negative* elbo
    #----------------------------------------------------

    options = Optim.Options(extended_trace = false, store_trace = false, show_every = 1, show_trace = false,  iterations = iterations, g_tol = 1e-6, callback = trackELBO)

    result  = Optim.optimize(minauxiliary, gradhelper, [μ₀; Cdiag], optimiser, options)

    μopt, Copt = unpack(result.minimizer)


    #----------------------------------------------------
    # Return results
    #----------------------------------------------------

    Σopt = getcov(Copt)

    return MvNormal(μopt, Σopt), elbo(μopt, Copt, Ztrain), Copt

end
