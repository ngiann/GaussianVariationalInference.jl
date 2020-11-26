entropy(C::Array{T, 2} where T<:Real) = 0.5*logdet(2*π*ℯ*C*C')

entropy(C::AbstractArray{T, 1} where T<:Real) = 0.5*sum(log.(C.^2)) + 0.5*log(2*π*ℯ) * length(C)

entropy(σ::Real) = 0.5*log(2*π*ℯ*σ^2)
