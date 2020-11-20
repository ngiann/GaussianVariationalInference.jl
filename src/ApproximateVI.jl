module ApproximateVI

    using Plots, Crayons.Box

    using LinearAlgebra, Random, Optim, ForwardDiff

    using Printf, ProgressMeter

    using Distributions, Statistics

    export VI, VIdiag

    include("util/plot_ellipse.jl")
    include("util/util.jl")
    include("util/entropy.jl")
    include("VI.jl")
    include("VIdiag.jl")

    gr()
end
