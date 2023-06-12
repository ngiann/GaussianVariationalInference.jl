"""
    Synthetic two-dimensional problem for verifying transform
    
# Example
```julia-repl
julia> logp, sample, f, g = verificationexample3() # target distribution to approximate
julia> q, logev = VI(logp, randn(2), S = 100, iterations = 10_000, show_every = 50, transform = f)
julia> using Plots # must be indepedently installed.
julia> x = 0.01:0.02:5.99
julia> contour(x, x, map(x -> exp(logp(collect(x))), Iterators.product(x, x))', fill=true, c=:blues, colorbar = false) # plot target
julia> contour!(x, x, map(x -> pdf(q,(collect(x))), Iterators.product(x, x))', color="red", alpha=0.3) # plot approximation q
julia> X = reduce(hcat, [f.(rand(q)) for i in 1:300]) 
julia> Plots.scatter!(X[1,:],X[2,:], markersize=2, alpha=0.2)
```
"""
function verificationexample3_diag()
   
    g(x) = invtransformbetween.(x, 0, 6)

    f(x) = transformbetween.(x, 0, 6)

    basedensity = MvNormal([1.0; 3.0], [0.5 0.0; 0.0 1.0])

    jac(x) = ForwardDiff.jacobian(x -> g(x), x)

    log_transformeddensity =  x -> logpdf(basedensity, g(x)) + logabsdet(jac(x))[1]

    sample() = f(rand(basedensity))
    
    return log_transformeddensity, sample, f, g
    
end
