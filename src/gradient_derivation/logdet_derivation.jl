function logdet_derivation(; seed = 1)

    D = 3

    rg = MersenneTwister(seed)

    C = randn(rg, D, D)

    u, v = randn(rg, D), randn(rg, D)

    makeroot(u,v) = C + u*v'
    
    function makecov(u, v) 
        local Σroot = makeroot(u, v)
        Σroot*Σroot'
    end

    f(u,v) = 0.5 * logdet(makecov(u, v))

    adgradu = ForwardDiff.gradient(x -> f(x, v), u)

    angradu = pinv(makeroot(u, v)') * v

    angradualt  = ((makeroot(u, v)') \ v)
    

    display([adgradu angradu angradualt])


    adgradv = ForwardDiff.gradient(x -> f(u, x), v)

    angradv = (makeroot(u, v) \ u)

    display([adgradv angradv])
end