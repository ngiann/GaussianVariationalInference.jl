function generatelatentZ(; S = S, D = D, seed=1)

    rg = MersenneTwister(seed)

    Z  = [randn(rg, D) for s in 1:S]

    return Z

end
