    function transformnormal(basedensity, g)

        jac(x) = ForwardDiff.jacobian(g, x)

        x-> logpdf(basedensity, g(x)) + logabsdet(jac(x))[1]

    end