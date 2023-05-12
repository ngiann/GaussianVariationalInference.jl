"""
Approximates unnormalised posterior with Gaussian distribution
"""
module GaussianVariationalInference

    using Transducers

    using ArgCheck, Printf, Crayons

    using LinearAlgebra, Optim
    
    using Random, Distributions, StatsFuns

    using Zygote, ForwardDiff
    
    
    # Core code
    
    include("interface.jl")
    include("VIfull.jl")
    include("VIdiag.jl")
    include("VIrank1.jl")
    include("elbo.jl")
    include("entropy.jl")
    
    export VI, VIdiag, VIrank1
    

    # Utilities
    
    include("util/pickoptimiser.jl")
    include("util/generatelatentZ.jl")
    include("util/defaultgradient.jl")
    include("util/verifygradient.jl")
    include("util/RecordELBOProgress.jl")
    include("util/makeparameters.jl")


    # Verification

    # include("gradient_derivation/logdet_derivation.jl")

    # export logdet_derivation


    # Example problems
    
    include("Examples/exampleproblem1.jl")

    export exampleproblem1
    

    # Re-export 
    
    export cov, mean, pdf, logpdf

end

