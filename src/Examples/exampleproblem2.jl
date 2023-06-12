"""
    Synthetic two-dimensional problem for verifying mean field variational inference under change of variables
    
# Example
```julia-repl
julia> logp, sample, f, g, basedensity = exampleproblem2() # target distribution to approximate
julia> q, logev = VIdiag(logp, randn(2), S = 1000, iterations = 10_000, show_every = 50, transform = f)
julia> display(basedensity); display(q); # these two densities must be close to each other
julia> using LinearAlgebra
julia> using Plots, ForwardDiff # must be indepedently installed.
julia> x = 0.001:0.02:5
julia> contourf(x, x, map(x -> exp(logp(collect(x))), Iterators.product(x, x))', c=:blues, colorbar = false) # plot target
julia> log_transformedq = x-> logpdf(q, g(x)) + logabsdet(ForwardDiff.jacobian(g, x))[1]
julia> contour!(x, x, map(x -> exp(log_transformedq(collect(x))), Iterators.product(x, x))', c=:red, colorbar = false) # plot target
```
"""
function exampleproblem2()

    Σdiag = [2.0 0.0; 0.0 1] # diagonal matrix!
    
    μ = [3.1; 0.2]

    g(y) = log.(y)

    f(x) = exp.(x)

    basedensity = MvNormal(μ, Σdiag)

    jac(x) = ForwardDiff.jacobian(g, x)

    log_transformeddensity =  x-> logpdf(basedensity, g(x)) + logabsdet(jac(x))[1]

    sample() = f(rand(basedensity))
    
    return log_transformeddensity, sample, f, g, basedensity
    
end