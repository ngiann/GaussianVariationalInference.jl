#----------------------------------------------------------------------------
function laplace(D::Int, logposterior::Function; show_trace=false)
#----------------------------------------------------------------------------

   laplace(randn(D), logposterior; show_trace=show_trace)

end

#----------------------------------------------------------------------------
function laplace(w0::Vector, logposterior::Function; show_trace=false)
#----------------------------------------------------------------------------

   gradlogposterior(w) = Zygote.gradient(logposterior, w)[1]

   gradlogposterior(storage, w) = copyto!(storage, gradlogposterior(w))

   laplace(w0, logposterior, gradlogposterior; show_trace=show_trace)

end

#----------------------------------------------------------------------------
function laplace(w0::Vector, logposterior::Function, gradlogposterior::Function; show_trace=false)
#----------------------------------------------------------------------------

   coreLaplace(w0, logposterior, gradlogposterior; show_trace=show_trace)

end

#----------------------------------------------------------------------------
function coreLaplace(w0::Vector, logposterior::Function, gradlogposterior::Function; show_trace=false)
#----------------------------------------------------------------------------

   negative_logposterior = x-> -1.0 * logposterior(x)

   function negative_logposterior_gradient!(storage, w)
        gradlogposterior(storage, w)
        storage .*= -1.0
        nothing
   end

   #  # find mode μ, we are *minimising*
   #  μ = Optim.optimize(negative_logposterior, negative_logposterior_gradient!, w0, LBFGS(),
   #          Optim.Options(iterations=100_000, show_trace=show_trace, g_tol=1e-6)).minimizer

   μ = mean(VIdiag(logposterior, w0; S = 1000, iterations=1000, show_every=10, gradientmode=:provided, gradlogp = gradlogposterior)[1])


    # Covariance of Gaussian posterior at mode μ is the inverse of Hessian
    Σ = get_covariance_at_mode(μ, negative_logposterior)

    # return Gaussian parameters
    return μ, Σ

end


#----------------------------------------------------------------------------
function get_covariance_at_mode(mode, negative_logposterior)
#----------------------------------------------------------------------------

   # find Hessian at mode
   H = Zygote.hessian(negative_logposterior, mode)

   # Covariance of Gaussian posterior at mode is the inverse of Hessian
   Σ = H \ I

   # Enforce symmetry
   return (Σ + Σ') / 2

end

#----------------------------------------------------
function eigendecomposition(Σ)
#----------------------------------------------------
   
       # decompose covariance
       EigenDecomposition = eigen(Σ)
   
       # get root of eigenvalues
       Esqrt = sqrt.(max.(EigenDecomposition.values, 1e-8))
   
       # get eigenvectors that stay fixed throughout
       V     = EigenDecomposition.vectors
   
       return V, Esqrt
   
   end