"""
    Synthetic two-dimensional problem
    
# Example
```julia-repl
julia> logp = exampleproblem1() # target distribution to approximate
julia> q, logev = VI(logp, randn(2), S = 100, iterations = 10_000, show_every = 50)
julia> using Plots # must be indepedently installed.
julia> x = -3:0.02:3
julia> contour(x, x, map(x -> exp(logp(collect(x))), Iterators.product(x, x))', fill=true, c=:blues, colorbar = false) # plot target
julia> contour!(x, x, map(x -> pdf(q,(collect(x))), Iterators.product(x, x))', color="red", alpha=0.3) # plot approximation q
```
"""
function exampleproblem1()

    w(x)    = sin(2π*x/4)

    U(z)    = (z[2]-w(z[1]))^2

    loglikel(z) = -U(z)

    logprior(z) = Distributions.logpdf(MvNormal(zeros(2), 1.0), z)

    logp(z) = loglikel(z) + logprior(z)

    return logp
    
end
