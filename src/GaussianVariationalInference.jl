"""
Approximates unnormalised posterior with Gaussian distribution
"""
module GaussianVariationalInference

    using Transducers

    using ArgCheck, Printf, Crayons

    using LinearAlgebra, Random, Optim, ForwardDiff, Distributions

    using Zygote
    
    # Core code
    
    include("interface.jl")
    include("VIfull.jl")
    include("VIdiag.jl")
    include("VIrank1.jl")
    include("entropy.jl")
       
    # include("VIdiag.jl")
    # include("VIfixedcov.jl")
    # include("MVI.jl")
    # include("laplace.jl")
     
    
    # Utilities
    
    # include("util/report.jl")
    include("util/pickoptimiser.jl")
    include("util/generatelatentZ.jl")
    include("util/defaultgradient.jl")
    include("util/verifygradient.jl")
    include("util/RecordELBOProgress.jl")

    # Verification
    include("gradient_derivation/logdet_derivation.jl")


    # Example problems
    
    include("Examples/exampleproblem1.jl")

    

    export VI, VIdiag, VIrank1 #, VIdiag, VIfixedcov, MVI, laplace
    
    export exampleproblem1

    export logdet_derivation

    # Re-export 
    export cov, mean, pdf, logpdf
end

