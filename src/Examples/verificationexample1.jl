"""
    Synthetic two-dimensional problem for verifying transform
    
# Example
```julia-repl
julia> logp, f, g = verificationexample1() # target distribution to approximate
julia> q, logev = VI(logp, randn(2), S = 100, iterations = 10_000, show_every = 50, transform = f)
julia> using Plots # must be indepedently installed.
julia> x = -3:0.02:3
julia> contour(x, x, map(x -> exp(logp(collect(x))), Iterators.product(x, x))', fill=true, c=:blues, colorbar = false) # plot target
julia> contour!(x, x, map(x -> pdf(q,(collect(x))), Iterators.product(x, x))', color="red", alpha=0.3) # plot approximation q
```
"""
function verificationexample1()

    A = cholesky([2.0 0.6; 0.6 1]).L
    
    b = [1; -3.0]

    g(y) = A\(y-b)

    f(x) = A*x + b

    basedensity = MvNormal(zeros(2), 1.0)

    jac(x) = ForwardDiff.jacobian(g, x)

    log_transformeddensity =  x-> logpdf(basedensity, g(x)) + logabsdet(jac(x))[1]

    return log_transformeddensity, f, g
    
end
