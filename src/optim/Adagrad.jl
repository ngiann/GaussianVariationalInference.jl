mutable struct Adagrad
    α # learning rate
    ε # small value
    s # sum of squared gradient
end


# function Adagrad(α, ϵ, D)

# 	Adagrad(α, ϵ, zeros(D))

# end

function step!(M::Adagrad, x, g)
    α, ε, s = M.α, M.ε, M.s
    s[:] += g.*g
    return x + α*g ./ (sqrt.(s) .+ ε)
end