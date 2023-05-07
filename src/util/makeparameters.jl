makeparam(μ, C::Matrix, z) = μ + C*z

makeparam(μ, C::Vector, z) = μ + C.*z

# alias
makeparameter = makeparam
