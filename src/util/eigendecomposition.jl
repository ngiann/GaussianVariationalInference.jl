function eigendecomposition(Σ)
    
    # decompose covariance
    EigenDecomposition = eigen(Σ)

    # get root of eigenvalues
    Esqrt = sqrt.(max.(EigenDecomposition.values, 1e-8))

    # get eigenvectors that stay fixed throughout
    V     = EigenDecomposition.vectors

    return V, Esqrt

end
