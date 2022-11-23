"""
    Synthetic two-dimensional problem
    
# Example
```julia-repl
julia> logp = exampleproblem1() # Target distribution to approximate
julia> q, logev = VI(logp, randn(2), S = 100, iterations = 100, show_every = 5)
julia> using PyPlot # PyPlot, or any other plotting package, must be indepedently installed.
julia> x=-3:0.02:3
julia> pcolor(x, x, map(x -> exp(logp(collect(x))), Iterators.product(x, x))') # plot target
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
