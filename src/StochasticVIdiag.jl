function corestochVIdiag(logp::Function, μ₀::Vector, Cdiag::Vector; gradlogp = gradlogp, seed = seed, S = S, iterations = iterations, numerical_verification = numerical_verification, show_every = show_every, η = η, threshold = threshold)

    D = length(μ₀)

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

    function minauxiliary(param, Z)

        local μ, C = unpack(param)

        local ℓ, = elbo(μ, C, Z)

        return -1.0 * ℓ # we are minimising!

    end


    function minauxiliary_grad(param, Z)

        local μ, C = unpack(param)
        
        return -1.0 * elbo_grad(μ, C, Z)  # we are minimising!

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
    # Call optimiser to minimise *negative* elbo
    #----------------------------------------------------
    
    opt = Adadelta(1e-7, 0.95, 2D)

    θ = [μ₀; Cdiag]

    for iter in 1:iterations

        Z = [randn(D) for _ in 1:S]
        
        g = minauxiliary_grad(θ, Z)

        θ = step!(opt, θ, g)

        if mod(iter, show_every) == 1
        
            @printf("Adadelta: iter %d,\t ELBO ≈ %f\r",iter, -minauxiliary(θ, [randn(D) for _ in 1:10*S]))
        
        elseif iter == iterations
            
            @printf("\n")

        end

    end
   

    #----------------------------------------------------
    # Return results
    #----------------------------------------------------
	
    return unpack(θ)

end
