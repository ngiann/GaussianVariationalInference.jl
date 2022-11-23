module ApproximateVI

    using Transducers#ThreadsX

    using ArgCheck, Printf, Crayons

    using LinearAlgebra, Random, Optim, ForwardDiff, Distributions


    # Core code
    
    include("interface.jl")
    include("VIfull.jl")
    include("entropy.jl")
       
    # include("VIdiag.jl")
    # include("VIfixedcov.jl")
    # include("MVI.jl")
    # include("laplace.jl")
     
    
    # Utilities
    
    include("util/report.jl")
    include("util/generatelatentZ.jl")
    include("util/defaultgradient.jl")
    include("util/verifygradient.jl")


    # Example problems
    
    include("Examples/exampleproblem1.jl")

    

    export VI #, VIdiag, VIfixedcov, MVI, laplace
    
    export exampleproblem1
end

