defaultgradient(D::Int) = x -> NaN * ones(D)

defaultgradient(μ::Array{T, 1}) where T = x -> NaN * ones(length(μ))