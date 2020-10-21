function entropy(Σ::AbstractArray)

    0.5*logdet(2*π*ℯ*Σ)

end

function entropy(σ::Real)

    0.5*log(2*π*ℯ*σ^2)

end




function entropy_sqrt_eigenvalues(Esqrt)

    local D = length(Esqrt)

    local H = 0.5*D*log(2*π*2.7182818)

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
