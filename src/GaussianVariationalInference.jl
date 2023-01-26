"""
Approximates unnormalised posterior with Gaussian distribution
"""
module GaussianVariationalInference

    using Transducers

    using ArgCheck, Printf, Crayons

    using LinearAlgebra, Random, Optim, ForwardDiff, Distributions

    using MiscUtil

    # Core code
    
    include("interface.jl")
    include("VIfull.jl")
    include("entropy.jl")
       
    # include("VIdiag.jl")
    # include("VIfixedcov.jl")
    # include("MVI.jl")
    # include("laplace.jl")
     
    
    # Utilities
    
    # include("util/report.jl")
    include("util/generatelatentZ.jl")
    include("util/defaultgradient.jl")
    include("util/verifygradient.jl")
    include("util/RecordELBOProgress.jl")


    # Example problems
    
    include("Examples/exampleproblem1.jl")
    include("Examples/exampleproblem2.jl")


    # Verification problems for debugging purposes

    include("Examples/verificationexample1.jl")
    include("Examples/verificationexample2.jl")
    include("Examples/verificationexample3.jl")

    export VI #, VIdiag, VIfixedcov, MVI, laplace
    
    export exampleproblem1, exampleproblem2, verificationexample1, verificationexample2, verificationexample3

    # Re-export 
    export cov, mean, pdf, logpdf
end

