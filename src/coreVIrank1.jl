function coreVIrank1(logp::Function, μ₀::AbstractArray{T, 1}, C₀::AbstractArray{T, 2}; gradlogp = gradlogp, seed = seed, S = S, test_every = test_every, optimiser = optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, transform = transform) where T

    D = length(μ₀); @assert(D == size(C₀, 1) == size(C₀, 2))

    #----------------------------------------------------
    # generate latent variables
    #----------------------------------------------------

    Ztrain = generatelatentZ(S = S, D = D, seed = seed)


    #----------------------------------------------------
    # Define jacobian of transformation via AD
    #----------------------------------------------------

    jac_transform = transform == identity ? Matrix(I, D, D) : x -> ForwardDiff.jacobian(transform, x)


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

        local C = getcovroot(C₀, u, v)

        local ℓ = elbo(μ, C, Ztrain)

        update!(trackELBO; newelbo = ℓ, μ = μ, C = C)

        return -1.0 * ℓ # Optim.optimise is minimising

    end


    function minauxiliary_grad(param)

        local μ, u, v = unpack(param)

        return -1.0 * elbo_grad(μ, u, v, Ztrain)  # Optim.optimise is minimising

    end


    #----------------------------------------------------
    # Functions for covariance and covariance root 
    #----------------------------------------------------

    function getcov(C₀, u, v)

        local aux = getcovroot(C₀, u, v)

        local Σ = aux*aux'

        (Σ + Σ') / 2 # for numerical stability

    end


    function getcovroot(C₀, u, v)
    
        C₀ + u*v'

    end


    #----------------------------------------------------
    # Approximate evidence lower bound and its gradient
    #----------------------------------------------------

    function elbo(μ, C, Z)

        
        local ℋ = GaussianVariationalInference.entropy(C)
        
        # if transform !== identity
            
        #     local auxentropy = z -> logabsdet(jac_transform(μ .+ C*z))[1]
            
        #     ℋ += Transducers.foldxt(+, Map(auxentropy),  Z) / length(Z) 
            
        # end
        
        local auxexpectedlogl = z -> logp(transform(μ .+ C*z))

        local Elogl = Transducers.foldxt(+, Map(auxexpectedlogl),  Z) / length(Z)
        
        return Elogl + ℋ

    end


    function partial_elbo_grad(μ, C, u, v, z)

        local g = gradlogp(μ .+ C*z)

        local gμ = g

        local gu = g*v'*z

        local gv = u'*g*z
            
        [gμ; vec(gu); vec(gv)]
        
    end


    function elbo_grad(μ, u, v, Z)

        local  C = getcovroot(C₀, u, v)

        local aux = z -> partial_elbo_grad(μ, C, u, v, z)
        
        local gradμuv = Transducers.foldxt(+, Map(aux), Z) / length(Z)

        # entropy contribution to covariance

        local gu_entr = C' \ v
        
        gradμuv[D+1:2D] += gu_entr

        local gv_entr = C \ u
        
        gradμuv[2D+1:3D] += gv_entr
        
        return gradμuv

    end


    # Package Optim requires that function for gradient has following signature

    gradhelper(storage, param) = copyto!(storage, minauxiliary_grad(param))


    #----------------------------------------------------
    # Numerically verify gradient
    #----------------------------------------------------

    # COMMENT BACK IN AFTER VERIFICATION
    #numerical_verification ? verifygradient(μ₀, Σ₀, elbo, minauxiliary_grad, unpack, Ztrain) : nothing
    
    # DELETE AFTER VERIFICATION
    # let 
        
    #     local u,v = randn(D), randn(D)

    #     local angrad = minauxiliary_grad([μ₀;vec(u);vec(v)])
        
    #     adgrad = ForwardDiff.gradient(minauxiliary, [μ₀; vec(u);vec(v)])
    
    #     discrepancy =  maximum(abs.(vec(adgrad) - vec(angrad)))
    
    #     display([angrad adgrad])

    #     @printf("Maximum absolute difference between AD and analytical gradient is %f\n", discrepancy)
        
    # end

    #----------------------------------------------------
    # Define callback function called at each iteration
    #----------------------------------------------------

    # We want to keep track of the best variational 
    # parameters encountered during the optimisation of
    # the elbo. Unfortunately, the otherwise superb
    # package Optim.jl does not provide a consistent way
    # accross different optimisers to do this.

    
    trackELBO = RecordELBOProgress(; μ = zeros(D), C = zeros(D,D), 
                                     Stest = Stest,
                                     show_every = show_every,
                                     test_every = test_every, 
                                     elbo = elbo, seed = seed)
    


    #----------------------------------------------------
    # Call optimiser to minimise *negative* elbo
    #----------------------------------------------------

    options = Optim.Options(extended_trace = false, store_trace = false, show_trace = false,  iterations = iterations, g_tol = 1e-6, callback = trackELBO)

    result  = Optim.optimize(minauxiliary, gradhelper, [μ₀; 0.01*randn(2D)], LBFGS(), options)

    μopt, uopt, vopt = unpack(result.minimizer)


    #----------------------------------------------------
    # Return results
    #----------------------------------------------------

    Copt = getcovroot(C₀, uopt, vopt)

    Σopt = getcov(C₀, uopt, vopt)

    return μopt, Copt, elbo(μopt, Copt, Ztrain)

end
