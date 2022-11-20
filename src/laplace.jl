function coreLaplace(logposterior::Function,
                 gradlogposterior::Function,
              hessianlogposterior::Function,
                               w0::Array{Float64,1};
                       iterations = iterations,
                        optimiser = optimiser,
                       show_every = show_every)

    @printf("Running laplace with D=%d for %d iterations\n", length(w0), iterations)


    negative_logposterior = x-> -1.0 * logposterior(x)

    negative_logposterior_gradient!(storage, w) = copyto!(storage, -1.0 * gradlogposterior(w))


    # find mode μ, we are *minimising*
    μ = Optim.optimize(negative_logposterior, negative_logposterior_gradient!, w0, optimiser,
            Optim.Options(iterations=iterations, show_trace=true, show_every=show_every)).minimizer

    # Covariance of Gaussian posterior at mode μ is the inverse of Hessian of negative log-likelihood
    Σ = get_covariance_at_mode(μ, x->-1*hessianlogposterior(x))

    # return Gaussian parameters
    return MvNormal(μ, Σ)

end



######################################################################
function get_covariance_at_mode(mode, hessianlogposterior)
######################################################################

    # find Hessian at mode
    H = hessianlogposterior(mode)

    # Covariance of Gaussian posterior at mode is the inverse of Hessian
    Σ = try

        ((H+H')/2)\I

    catch err
        if isa(err, LinearAlgebra.LAPACKException)
            local U,S,V = svd(H)
            V*Diagonal(1.0./max.(S, 1e-6))*U'
        else
            throw(err)
        end
    end


    # Enforce symmetry
    Σ = (Σ + Σ') * 0.5 + 1e-8*I

    return Σ

end
