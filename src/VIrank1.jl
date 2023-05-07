function coreVIrank1(logp::Function, μ₀::AbstractArray{T, 1}, C::AbstractArray{T, 2}; gradlogp = gradlogp, seed = seed, S = S, test_every = test_every, optimiser = optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, transform = transform, seedtest = seedtest) where T

    D = length(μ₀)

    rg = MersenneTwister(seed)


    #----------------------------------------------------
    # generate latent variables
    #----------------------------------------------------

    Ztrain = generatelatentZ(S = S, D = D, seed = seed)


    #----------------------------------------------------
    # Define jacobian of transformation via AD
    #----------------------------------------------------

    # jac_transform = transform == identity ? Matrix(I, D, D) : x -> ForwardDiff.jacobian(transform, x)


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

        local ℓ = elbo(μ, u, v, Ztrain)

        update!(trackELBO; newelbo = ℓ, μ = μ, C = getcovroot(u, v))

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


    function getcovroot(u, v)
    
        C + u*v'

    end


    #----------------------------------------------------
    # Approximate evidence lower bound and its gradient
    #----------------------------------------------------

    elbo(μ, u, v, Z) = elbo(μ, getcovroot(u, v), Z)

    function elbo(μ, C, Z)

        local ℋ = GaussianVariationalInference.entropy(C)
        
        # if transform !== identity
            
        #     local auxentropy = z -> logabsdet(jac_transform(μ .+ C*z))[1]
            
        #     ℋ += Transducers.foldxt(+, Map(auxentropy),  Z) / length(Z) 
            
        # end
        
        local auxexpectedlogl = z -> logp(transform(makeparameter(μ, C, z)))

        local Elogl = Transducers.foldxt(+, Map(auxexpectedlogl),  Z) / length(Z)
        
        return Elogl + ℋ

    end


    function partial_elbo_grad(μ, C, u, v, z)

        local g = gradlogp(makeparameter(μ, C, z))

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

    numerical_verification ? verifygradient(μ₀, 1e-2*randn(rg, D), 1e-2*randn(rg, D), elbo, minauxiliary_grad, unpack, Ztrain) : nothing

    
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
                                     elbo = elbo, seed = seedtest)
    


    #----------------------------------------------------
    # Call optimiser to minimise *negative* elbo
    #----------------------------------------------------

    options = Optim.Options(extended_trace = false, store_trace = false, show_trace = false,  iterations = iterations, g_tol = 1e-6, callback = trackELBO)

    result  = Optim.optimize(minauxiliary, gradhelper, [μ₀; 1e-2*randn(rg, 2D)], optimiser, options)

    μopt, uopt, vopt = unpack(result.minimizer)


    #----------------------------------------------------
    # Return results
    #----------------------------------------------------

    Copt = getcovroot(uopt, vopt)

    return MvNormal(μopt, getcov(uopt, vopt)), elbo(μopt, uopt, vopt, Ztrain), Copt

end
