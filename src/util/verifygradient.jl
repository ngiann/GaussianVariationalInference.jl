# Checks gradient for VI with full covariance matrix
function verifygradient(μ, Σ::Matrix, elbo, minauxiliary_grad, unpack, Z)
            
    C = vec(Matrix(cholesky(Σ).L))
   
    angrad = minauxiliary_grad([μ; C])
    
    adgrad = ForwardDiff.gradient(p -> -elbo(unpack(p)..., Z), [μ;  C])

    reportdiscrepancy(angrad, adgrad)

end


# Checks gradient for VI with diagonal covariance matrix
function verifygradient(μ, Cdiag::Vector, elbo, minauxiliary_grad, unpack, Z)

    angrad = minauxiliary_grad([μ; Cdiag])
    
    adgrad = ForwardDiff.gradient(p -> -elbo(unpack(p)..., Z), [μ; Cdiag])

    reportdiscrepancy(angrad, adgrad)

end


# Checks gradient for VI with rank 1 parametrised covariance matrix
function verifygradient(μ, u::Vector, v::Vector, elbo, minauxiliary_grad, unpack, Z)

    angrad = minauxiliary_grad([μ; u; v])
    
    adgrad = ForwardDiff.gradient(p -> -elbo(unpack(p)..., Z), [μ; u; v])

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