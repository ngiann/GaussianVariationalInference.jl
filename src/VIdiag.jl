function coreVIdiag(logp::Function, μ₀::Vector, C₀diag::Vector; gradlogp = gradlogp, seed = seed, S = S, test_every = test_every, optimiser = optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, parallelmode = parallelmode, transform = transform)

    D = length(μ₀)

    #----------------------------------------------------
    # generate latent variables
    #----------------------------------------------------

    Ztrain = generatelatentZ(S = S, D = D, seed = seed)


    #----------------------------------------------------
    # Define jacobian of transformation via AD
    #----------------------------------------------------

    # jac_transform   = transform == identity ? Matrix(I, D, D) : x -> ForwardDiff.jacobian(transform, x)

    jac_transform = transform == identity ? ones(D) : x -> firstderivative.(transform, x)
    
    firstderivativetransform  = transform == identity ? x -> one(eltype(x))  : firstderivativefunction(transform)

    secondderivativetransform = transform == identity ? x -> zero(eltype(x)) : secondderivativefunction(transform)


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


    #----------------------------------------------------
    # Functions for covariance and covariance root 
    #----------------------------------------------------

    getcov(Cdiag) = Diagonal(Cdiag.^2)
    
    getcovroot(Cdiag) = Cdiag



    #----------------------------------------------------
    # Approximate evidence lower bound and its gradient
    #----------------------------------------------------

    function elbo(μ, Cdiag, Z)

        local ℋ = GaussianVariationalInference.entropy(Cdiag)
        
        if transform !== identity
            
            auxentropy = z -> sum(log.(abs.(jac_transform.(makeparam(μ, Cdiag, z))))) 

            ℋ += evaluatesamples(auxentropy, Z, Val(parallelmode))
            
        end
  
        local auxexpectedlogl = z -> logp(transform(makeparam(μ, Cdiag, z)))

        local Elogl = evaluatesamples(auxexpectedlogl, Z, Val(parallelmode))
        
        return Elogl + ℋ

    end


    function partial_elbo_grad(μ, Cdiag, z)

        local ψ = makeparameter(μ, Cdiag, z)

        local g = gradlogp(transform(ψ))

        # The following two lines could be left out when `transform`` is equal to `identity`

        g = g .* firstderivativetransform.(ψ) # contribution of transformation to data log-likelihood

        g += (1.0./firstderivativetransform.(ψ)) .* secondderivativetransform.(ψ) # contribution of transformation to entropy

        [g; vec(g.*z)]

    end


    function elbo_grad(μ, Cdiag, Z)
       
        local aux = z -> partial_elbo_grad(μ, Cdiag, z)
        
        local gradμCdiag = evaluatesamples(aux, Z, Val(parallelmode))

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

    options = Optim.Options(extended_trace = false, store_trace = false, show_every = 1, show_trace = false, iterations = iterations, g_tol = 1e-6, callback = trackELBO)

    result  = Optim.optimize(minauxiliary, gradhelper, [μ₀; C₀diag], optimiser, options)

    μopt, Copt = unpack(result.minimizer)


    #----------------------------------------------------
    # Return results
    #----------------------------------------------------

    return MvNormal(μopt, getcov(Copt)), elbo(μopt, Copt, Ztrain), Copt

end
