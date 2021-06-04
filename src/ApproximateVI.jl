module ApproximateVI

    using PyPlot

    using LinearAlgebra, Random, Optim, ForwardDiff

    using Printf, ProgressMeter

    using Distributions

    export VI, VIdiag, VIsphere, MVI, laplace

    include("util/plot_ellipse.jl")
    include("util/util.jl")
    include("util/entropy.jl")
    include("VIcalls.jl")
    include("VIfull.jl")
    include("VIdiag.jl")
    include("VIfixedcov.jl")
    include("MVI.jl")
    include("laplace.jl")

end
