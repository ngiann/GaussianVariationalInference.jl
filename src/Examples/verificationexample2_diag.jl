"""
    Synthetic two-dimensional problem for verifying transform
    
# Example
```julia-repl
julia> logp, sample, f, g = verificationexample2() # target distribution to approximate
julia> q, logev = VI(logp, randn(2), S = 100, iterations = 10_000, show_every = 50, transform = f)
julia> using Plots # must be indepedently installed.
julia> x = 0.001:0.02:5
julia> contour(x, x, map(x -> exp(logp(collect(x))), Iterators.product(x, x))', fill=true, c=:blues, colorbar = false) # plot target
julia> contour!(x, x, map(x -> pdf(q,(collect(x))), Iterators.product(x, x))', color="red", alpha=0.3) # plot approximation q
```
"""
function verificationexample2_diag()

    Σdiag = [2.0 0.0; 0.0 1]
    
    μ = [0.9; 0.2]

    g(y) = log.(y)

    f(x) = exp.(x)

    basedensity = MvNormal(μ, Σdiag)

    jac(x) = ForwardDiff.jacobian(g, x)

    log_transformeddensity =  x-> logpdf(basedensity, g(x)) + logabsdet(jac(x))[1]

    sample() = f(rand(basedensity))
    
    return log_transformeddensity, sample, f, g
    
end
