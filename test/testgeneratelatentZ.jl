function testgeneratelatentZ()

    Z = GaussianVariationalInference.generatelatentZ(S = 13, D = 3, seed = 101)

    @test length(Z)    == 13
    @test length(Z[1]) == 3

    Z1 = GaussianVariationalInference.generatelatentZ(S = 13, D = 3, seed = 101)

    @test Z1[1] == Z[1]

end