function importancesampling(logp, q, transform = identity; numsamples = 100, seed = 1)

    rg = MersenneTwister(seed)

    samples = [transform(rand(rg, q)) for i in 1:numsamples]

    logweights = map(s -> logp(s) - logpdf(q, s), samples)
    
    P = Categorical(exp.(logweights .- logsumexp(logweights)))

    sampler()  = samples[rand(P)]
    
    sampler(N) = samples[rand(P, N)]

    return sampler

end