# Checks gradient for VI with full covariance matrix
function verifygradient(μ, C::Matrix, elbo, minauxiliary_grad, unpack, Z)

    angrad = minauxiliary_grad([μ; vec(C)])
    
    adgrad = ForwardDiff.gradient(p -> -elbo(unpack(p)..., Z)[1], [μ;  vec(C)])

    reportdiscrepancy(angrad, adgrad)

end


# Checks gradient for VI with diagonal covariance matrix
function verifygradient(μ, Cdiag::Vector, elbo, minauxiliary_grad, unpack, Z)

    angrad = minauxiliary_grad([μ; Cdiag])
    
    adgrad = ForwardDiff.gradient(p -> -elbo(unpack(p)..., Z)[1], [μ;  Cdiag])

    reportdiscrepancy(angrad, adgrad)

end


# Checks gradient for VI with rank 1 parametrised covariance matrix
function verifygradient(μ, u::Vector, v::Vector, elbo, minauxiliary_grad, unpack, Z)

    angrad = minauxiliary_grad([μ; u; v])
    
    adgrad = ForwardDiff.gradient(p -> -elbo(unpack(p)..., Z)[1], [μ; u; v])

    reportdiscrepancy(angrad, adgrad)

end


function reportdiscrepancy(angrad, adgrad)

    discrepancy = maximum(abs.(vec(adgrad) - vec(angrad)))
    
    msg = @sprintf("Maximum absolute difference between AD and analytical gradient is %.8f\n", discrepancy)
    
    clr = discrepancy > 1e-5 ? :red : :cyan

    print(Crayon(foreground = clr, bold=true), msg)

    print(Crayon(foreground = :white, bold=false), "", Crayon(reset = true))

    nothing

end