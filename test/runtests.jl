using ApproximateVI
using Test

include("testentropy.jl")

@testset "ApproximateVI.jl" begin
    testentropy()
end
