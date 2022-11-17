defaultgradient(D::Int) where T = x -> NaN * ones(D)

defaultgradient(μ::Array{T, 1}) where T = x -> NaN * ones(length(μ))