entropy(C::Array{Float64, 2}) = 0.5*logdet(2*π*ℯ*C*C')

entropy(C::Array{Float64, 1}) = 0.5*sum(log.(C.^2)) + 0.5*log(2*π*ℯ) * length(C)

entropy(σ::Real) = 0.5*log(2*π*ℯ*σ^2)




function entropy_sqrt_eigenvalues(Esqrt)

    local D = length(Esqrt)

    local H = 0.5*D*log(2*π*ℯ)

    for esqrt in Esqrt
        H += 0.5*log(esqrt^2)
    end

    return H
end


function debug_entropy_sqrt_eigenvalues()

    dim = ceil(Int,1+rand()*30)
    A   = randn(11,11)
    Σ   = A*A'

    D = eigen(Σ)

    Distributions.entropy(MvNormal(zeros(11), Σ)),
    entropy_sqrt_eigenvalues(sqrt.(D.values))

end
