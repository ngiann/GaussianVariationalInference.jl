function importancesampling(logp, q, transform = identity; numsamples = 100, seed = 1)

    rg = MersenneTwister(seed)

    samples = [transform(rand(rg, q)) for i in 1:numsamples]

    logweights = map(s -> logp(s) - logpdf(q, s), samples)
    
    weights = exp.(logweights .- logsumexp(logweights))

    return samples, weights

end

function estimatecov(logp, q; numsamples = 100, seed = 1)

    samples, weights = importancesampling(logp, q; numsamples = numsamples, seed = seed)

    C = zeros(length(q), length(q))

    μ = mean(q)

    for i in 1:length(weights)

        C += w[i]*(samples[i] - μ)*(samples[i] - μ)'

    end

    return C

end