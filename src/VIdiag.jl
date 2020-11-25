"""
    VIdiag(logl, μ; S = 100, iterations = 1)

    VIdiag(logl, gradlogl, μ; S = 100, iterations = 1)

    VIdiag(logl, gradlogl, μ, Σ; S = 100, iterations = 1)

Returns mean and covariance of approximate diagonal Gaussian posterior inferred via approximate VI

"""
function VIdiag(logl::Function, μ::Array{Float64,1}; seed = 1, S = 100, iterations = 1, optimiser = Optim.LBFGS(), show_every = -1)

    gradlogl(x)   = ForwardDiff.gradient(logl, x)

    gradlogl(v,x) = ForwardDiff.gradient!(v, logl, x)

    VIdiag(logl, gradlogl, μ; seed = seed, S = S, iterations = iterations, optimiser = optimiser, show_every = show_every)

end


function VIdiag(logl::Function, gradlogl::Function, μ::Array{Float64,1}; seed = 1, S = 100, iterations = 1, optimiser=Optim.LBFGS(), show_every = -1)


    VIdiag(logl, gradlogl, μ, 0.1*ones(length(μ)); seed = seed, S = S, iterations = iterations, optimiser = optimiser, show_every = show_every)

end


function VIdiag(logl::Function, gradlogl::Function, μ::Array{Float64,1}, Σdiag::Array{Float64,1}; seed = 1, S = 100, iterations = 1, optimiser=Optim.LBFGS(), show_every = -1)


    coreVIdiag(logl, gradlogl, μ, Σdiag; seed = seed, S = S, iterations = iterations, optimiser = optimiser, show_every = show_every)

end



function coreVIdiag(logl::Function, gradlogl::Function, μ::Array{Float64,1}, Σdiag::Array{Float64,1}; seed = 1, S = 100, iterations = 1, optimiser=Optim.LBFGS(), show_every = -1)


    D = length(μ)

    @assert(D == length(Σdiag))

    @printf("Running VI diagonal with S=%d, D=%d for %d iterations\n", S, D, iterations)

    #----------------------------------------------------
    # Initialise matrix root
    #----------------------------------------------------

    Cdiag = sqrt.(Σdiag)


    #----------------------------------------------------
    # generate latent variables
    #----------------------------------------------------

    Ztrain = generatelatentZ(S = S, D = D, seed = seed)


    #----------------------------------------------------
    function unpack(param)
    #----------------------------------------------------

        @assert(length(param) == D+D)

        local μ = param[1:D]

        local Cdiag = reshape(param[D+1:D+D], D)

        return μ, Cdiag

    end


    #----------------------------------------------------
    function minauxiliary(param)
    #----------------------------------------------------

        local μ, Cdiag = unpack(param)

        return -1.0 * elbo(μ, Cdiag, Ztrain)

    end


    #----------------------------------------------------
    function minauxiliary_grad(param)
    #----------------------------------------------------

        local μ, Cdiag = unpack(param)

        return -1.0 * elbo_grad(μ, Cdiag, Ztrain)

    end


    #----------------------------------------------------
    function getcov(Cdiag)
    #----------------------------------------------------

        Cdiag.^2

    end


    #----------------------------------------------------
    function getcovroot(Cdiag)
    #----------------------------------------------------

        return Cdiag

    end


    #----------------------------------------------------
    function elbo(μ, Cdiag, Z)
    #----------------------------------------------------

        mean(map(z -> logl(μ .+ Cdiag.*z), Z)) + entropy(Cdiag)

    end


    #----------------------------------------------------
    function elbo_grad(μ, Cdiag, Z)
    #----------------------------------------------------

        local gradC = 1.0 ./ Cdiag # entropy contribution

        local gradμ = zeros(eltype(μ), D)

        local S     = length(Z)

        for s=1:S
            g      = gradlogl(μ .+ Cdiag .* Z[s])
            gradC += g .* Z[s] / S
            gradμ += g / S
        end

        return [vec(gradμ); gradC]

    end


    #----------------------------------------------------
    # Numerically verify gradient
    #----------------------------------------------------

    # adgrad = ForwardDiff.gradient(minauxiliary, [μ; Cdiag])
    # angrad = minauxiliary_grad([μ;Cdiag])
    # display([vec(adgrad) vec(angrad)])
    # display(maximum(abs.([vec(adgrad) vec(angrad)])))
    # return


    #----------------------------------------------------
    # Call optimiser
    #----------------------------------------------------

    gradhelper(storage, param) = copyto!(storage, minauxiliary_grad(param))

    opt    = Optim.Options(show_trace=true, show_every=show_every, g_tol=1e-6, iterations=iterations)

    result = Optim.optimize(x -> minauxiliary(x), gradhelper, [μ; Cdiag], optimiser, opt)

    μopt, Copt = unpack(result.minimizer)


    #----------------------------------------------------
    # Return results
    #----------------------------------------------------

    negativelogevidence = -1.0 * result.minimum

    return MvNormal(μopt, Diagonal(getcov(Copt))), elbo(μopt, Copt, generatelatentZ(S = 10*S, D = D, seed = seed+2))

end
