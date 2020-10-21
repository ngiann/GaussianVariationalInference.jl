#----------------------------------------------------
function generatelatentZ(; S = S, D = D, seed=1)
#----------------------------------------------------

    rg = MersenneTwister(seed)

    Z  = [randn(rg, D) for s in 1:S]

    return Z

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




#-------------------------------------------------------
function elbo(; logl = logl, μ = μ, C = C, Z = Z)
#-------------------------------------------------------

    # check dimensions

    @assert(length(μ) == size(C, 1) == size(C, 2) == length(Z[1]))

    # number of drawn samples of latent variables Z

    S = length(Z)

    # Integrated log-likelihood

    Elogℓ = mean(map(z -> logl(μ .+ C*z), Z))

    # Gaussian entropy contribution

    H = 0.5*logdet(2*π*ℯ*(C*C'))

    # Elbo is the sum of integrated log-likelihood and entropy

     return Elogℓ + H
end
