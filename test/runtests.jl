using GaussianVariationalInference, LinearAlgebra, Distributions
using Test

include("testentropy.jl")

include("testgeneratelatentZ.jl")

@testset "GaussianVariationalInference.jl" begin

    testentropy()

    testgeneratelatentZ()

end
