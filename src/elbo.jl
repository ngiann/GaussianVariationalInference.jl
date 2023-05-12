 function elbo(logp::Function, μ, C, Z)

    f = z -> logp(makeparam(μ, C, z))

    logpsamples = Transducers.tcollect(Map(f),  Z)
    
    return mean(logpsamples) + entropy(C), sqrt(var(logpsamples)/length(Z))

end