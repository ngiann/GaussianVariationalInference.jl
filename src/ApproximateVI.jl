module ApproximateVI

    using Plots, Crayons.Box

    using LinearAlgebra, Random, Optim, ForwardDiff

    using Printf, ProgressMeter

    using Distributions, Statistics

    export VI


    include("util/util.jl")
    include("util/entropy.jl")
    include("VI.jl")

    gr()
end
