"""
    exampleproblem1()

    Synthetic 2D problem
    
"""
function exampleproblem1()

    w(x)    = sin(2π*x/4)

    U(z)    = (z[2]-w(z[1]))^2

    loglikel(z) = -U(z)

    logprior(z) = Distributions.logpdf(MvNormal(zeros(2), 1.0), z)

    logp(z) = loglikel(z) + logprior(z)

    return logp
    
end
