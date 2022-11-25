function coreVIfull(logp::Function, μ₀::Array{T, 1}, Σ₀::Array{T, 2}; gradlogp = gradlogp, seed = seed, S = S, test_every = test_every, optimiser = optimiser, iterations = iterations, numerical_verification = numerical_verification, Stest = Stest, show_every = show_every) where T

    D = length(μ₀)

    #----------------------------------------------------
    # generate latent variables
    #----------------------------------------------------

    Ztrain = generatelatentZ(S = S,     D = D, seed = seed)

    Ztest  = generatelatentZ(S = Stest, D = D, seed = seed + 1)


    #----------------------------------------------------
    # We want to keep track of the best variational 
    # parameters encountered during the optimisation of
    # the elbo. Unfortunately, the otherwise superb
    # package Optim.jl does not provide a consistent way
    # accross different optimisers to do this.
    #----------------------------------------------------
    
    bestμ, bestC, bestelbo = zeros(D), zeros(D, D), -Inf

    function updatebestsofar(ℓ, μ, C)

        if bestelbo < ℓ
            bestelbo = ℓ
            bestμ .= μ
            bestC .= C
        end

    end


    #----------------------------------------------------
    # Auxiliar function for handling parameters
    #----------------------------------------------------

    function unpack(param)

        @assert(length(param) == D+D*D)

        local μ = param[1:D]

        local C = reshape(param[D+1:D+D*D], D, D)

        return μ, C

    end


    #----------------------------------------------------
    # Objective and gradient functions for Optim.optimize
    #----------------------------------------------------

    function minauxiliary(param)

        local μ, C = unpack(param)

        local ℓ = elbo(μ, C, Ztrain)

        updatebestsofar(ℓ, μ, C)

        return -1.0 * ℓ # Optim.optimise is minimising

    end


    function minauxiliary_grad(param)

        local μ, C = unpack(param)

        return -1.0 * elbo_grad(μ, C, Ztrain)  # Optim.optimise is minimising

    end


    #----------------------------------------------------
    # Functions for covariance and covariance root 
    #----------------------------------------------------

    function getcov(C)

        local aux = C*C'

        (aux+aux') / 2 # for numerical stability

    end


    function getcovroot(C)
    
        return C

    end


    #----------------------------------------------------
    # Approximate evidence lower bound and its gradient
    #----------------------------------------------------

    function elbo(μ, C, Z)

        local aux = z -> logp(μ .+ C*z)

        Transducers.foldxt(+, Map(aux),  Z) / length(Z) + ApproximateVI.entropy(C)

    end


    function partial_elbo_grad(μ, C, z)

        local g = gradlogp(μ .+ C*z)
            
        [g; vec(g*z')] # gradμ = g, gradC = g*z'
        
    end


    function elbo_grad(μ, C, Z)
       
        local aux = z -> partial_elbo_grad(μ, C, z)
        
        local gradμC = Transducers.foldxt(+, Map(aux), Z) / length(Z)

        # entropy contribution to covariance

        gradμC[D+1:end] .+= vec((C\I)') 
        
        return gradμC

    end


    # Package Optim requires that function for gradient has following signature

    gradhelper(storage, param) = copyto!(storage, minauxiliary_grad(param))


    #----------------------------------------------------
    # Numerically verify gradient
    #----------------------------------------------------

    numerical_verification ? verifygradient(μ₀, Σ₀, elbo, minauxiliary_grad, unpack, Ztrain) : nothing
 

    #----------------------------------------------------
    # Define callback function called at each iteration
    #----------------------------------------------------

    # keep track of iterations and best elbo on test samples

    countiterations = 0
    
    currelbotest, prvelbotest = -Inf, -Inf
    

    function callback(_)
        
        countiterations += 1

        currelbotest, prvelbotest = report(countiterations, show_every, test_every, Stest, elbo, bestμ, bestC, Ztest, bestelbo, currelbotest, prvelbotest)
        
        return false

    end


    #----------------------------------------------------
    # Call optimiser to minimise *negative* elbo
    #----------------------------------------------------

    options = Optim.Options(extended_trace = false, store_trace = false, show_every = 1, show_trace = false,  iterations = iterations, g_tol = 1e-6, callback = callback)

    result  = Optim.optimize(minauxiliary, gradhelper, [μ₀; vec(cholesky(Σ₀).L)], optimiser, options)

    μopt, Copt = unpack(result.minimizer)


    #----------------------------------------------------
    # Return results
    #----------------------------------------------------

    Σopt = getcov(Copt)

    return MvNormal(μopt, Σopt), elbo(μopt, Copt, Ztrain)

end