function corestochasticVIrank1(logp::Function, μ₀::Vector, C₀::Matrix, u₀::Vector, v₀::Vector; gradlogp = gradlogp, seed = seed, S = S, test_every = test_every, optimiser = optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every, transform = transform, seedtest = seedtest, threshold = threshold)

    D = length(μ₀)

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

    function minauxiliary(param, Z)

        local μ, u, v = unpack(param)

        local ℓ, = elbo(μ, u, v, Z)

        return -1.0 * ℓ # Optim.optimise is minimising

    end


    function minauxiliary_grad(param, Z)

        local μ, u, v = unpack(param)

        return -1.0 * elbo_grad(μ, u, v, Z)  # Optim.optimise is minimising

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
    # Call optimiser to minimise *negative* elbo
    #----------------------------------------------------
    
    opt = Adadelta(1e-7, 0.95, 3D)

    θ = [μ₀; 1e-4*randn(2D)]

    for iter in 1:iterations

        Z = [randn(D) for _ in 1:S]
        
        g = minauxiliary_grad(θ, Z)

        θ = step!(opt, θ, g)

        if mod(iter, show_every) == 1
        
            @printf("%s: iter %d,\t ELBO ≈ %f\r", tostring(opt), iter, -minauxiliary(θ, [randn(D) for _ in 1:10*S]))
        
        elseif iter == iterations
            
            @printf("\n")

        end

    end
   

    #----------------------------------------------------
    # Return results
    #----------------------------------------------------
	
    return unpack(θ)


end
