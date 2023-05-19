function sgdmaximise(;w₀ = w₀, elbo = elbo, gradelbo = gradelbo, η = η, iterations = iterations, threshold = threshold, show_every = -1, seed = 1, S = 50)

    rg = MersenneTwister(seed)

    D = round(Int, length(w₀) / 2)

    w = copy(w₀)

    # prevelbo = elbo(w, [randn(rg, D) for s in 1:S])

    for i in 1:iterations

        Z = [randn(rg, D) for _ in 1:S]

        w = w + η * gradelbo(w, Z)

        if mod(i, show_every) == 0

            Ztest = [randn(rg, D) for _ in 1:10*S]

            @printf("SGDmaximise (%d): Iteration %5d, ELBO ≈ %f\r", S, i, elbo(w, Ztest)[1])

        end
        
    end

    w

end