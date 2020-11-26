function testentropy()

  # check my implementation of entropy against that of Distributions.jl

  D = 5
  A = randn(D, D)
  Σ = A*A'
  μ = randn(D)

  @test Distributions.entropy(MvNormal(μ, Σ)) ≈ ApproximateVI.entropy(A)

  D = 11
  A = randn(D)
  Σ = Diagonal(A.^2)
  μ = randn(D)

  @test Distributions.entropy(MvNormal(μ, Σ)) ≈ ApproximateVI.entropy(A)


  σ = rand()+1e-4
  μ = randn()

  @test Distributions.entropy(Normal(μ, σ)) ≈ ApproximateVI.entropy(σ)


end
