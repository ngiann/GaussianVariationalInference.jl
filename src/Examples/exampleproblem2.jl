"""
    Synthetic two-dimensional problem
    
# Example
```julia-repl
julia> logp, transform = exampleproblem2() # target distribution to approximate
julia> q, logev = VI(logp, randn(2), S = 100, iterations = 10_000, show_every = 10Z)
```
"""
function exampleproblem2(;seed = 1, N = 30)

    rg = MersenneTwister(seed)

    truedensity = Beta(1, 1)

    data = rand(rg, truedensity, N)

    function logp(param)

        local a, b = param

        mapreduce(x -> logpdf(Beta(a, b), x), +, data)

    end
   
    @printf("True density is\n"); display(truedensity)

    return logp, x -> [exp(x[1]); exp(x[2])]
    
end
