using ApproximateVI, LinearAlgebra, Distributions
using Test

include("testentropy.jl")

include("testgeneratelatentZ.jl")

@testset "ApproximateVI.jl" begin

    testentropy()

    testgeneratelatentZ()

end
