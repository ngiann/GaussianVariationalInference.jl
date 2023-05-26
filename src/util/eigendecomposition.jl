#----------------------------------------------------
function eigendecomposition(Σ)
#----------------------------------------------------

    # decompose covariance
    EigenDecomposition = eigen(Σ)

    # get root of eigenvalues
    Esqrt = sqrt.(max.(EigenDecomposition.values, 1e-3))

    # get eigenvectors that stay fixed throughout
    V     = EigenDecomposition.vectors

    return Matrix(1.0 * I,size(Σ,1),size(Σ,2)), Esqrt

end