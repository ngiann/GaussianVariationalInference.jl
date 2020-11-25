

using Plots, Crayons.Box

using LinearAlgebra, Random, Optim, ForwardDiff

using Printf, ProgressMeter

using Distributions, Statistics



include("util/plot_ellipse.jl")
include("util/util.jl")
include("util/entropy.jl")
include("VI.jl")
include("VIdiag.jl")

gr()
