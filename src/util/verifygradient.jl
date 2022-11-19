function verifygradient(μ₀, Σ₀, elbo, minauxiliary_grad, unpack, Ztrain)
            
    C = Matrix(cholesky(Σ₀).L)
    
    local angrad = minauxiliary_grad([μ₀;vec(C)])
    
    adgrad = ForwardDiff.gradient(p -> -elbo(unpack(p)..., Ztrain), [μ₀; vec(C)])

    discrepancy =  maximum(abs.(vec(adgrad) - vec(angrad)))

    msg = @sprintf("Maximum absolute difference between AD and analytical gradient is %f\n", discrepancy)
    
    clr = discrepancy > 1e-5 ? :red : :cyan

    print(Crayon(foreground = clr, bold=true), msg)

    print(Crayon(foreground = :white, bold=false), "")

    nothing

end