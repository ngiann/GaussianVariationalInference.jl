mutable struct RecordELBOProgress

    bestelbo # default -Inf
    bestelbo_std
    bestμ 
    bestC

    countiterations # default is 0

    Stest
    show_every
    test_every
    testelbofunction
    testelbohistory
    testelbohistory_std

end


function RecordELBOProgress(; μ = μ, C = C, Stest = Stest, show_every = show_every, test_every = test_every, logp = logp, seed = seed)
    
    D = length(μ)

    # Ztest = generatelatentZ(S = Stest, D = D, seed = 13)
    
    f = z -> logp(makeparam(μ, C, z))

    function testelbofunction(μ, C) 
        
        local aux = map(f, [randn(D) for _ in 1:100])

        while sqrt(var(aux)/length(aux)) > 0.1

            auxmore = Transducers.tcollect(Map(f),  [randn(D) for _ in 1:100])

            aux = vcat(aux, auxmore)

        end

        mean(aux) + entropy(C), sqrt(var(aux)/length(aux)), length(aux)

    end

    RecordELBOProgress(-Inf, -Inf, μ, C, 0, Stest, show_every, test_every, testelbofunction, zeros(Float64, 0), zeros(Float64, 0))

end


function update!(p::RecordELBOProgress; newelbo = newelbo, newelbo_std = newelbo_std, μ = μ, C = C)

    if newelbo > p.bestelbo
        p.bestelbo = newelbo
        p.bestelbo_std = newelbo_std
        p.bestμ .= μ
        p.bestC .= C
    end

end

function plot(p::RecordELBOProgress)

    figure(-1)
    cla()
    
    numentries = length(p.testelbohistory)

    PyPlot.plot(collect(1:numentries)*p.test_every, p.testelbohistory, "ko-")
    PyPlot.plot(collect(1:numentries)*p.test_every, p.testelbohistory + 3*p.testelbohistory_std, "k--")
    PyPlot.plot(collect(1:numentries)*p.test_every, p.testelbohistory - 3*p.testelbohistory_std, "k--")


end

function (p::RecordELBOProgress)(_) # used as callback


    p.countiterations += 1

    
    if p.countiterations == 1
        
        if p.show_every > 0

            @printf("Reporting elbo every %d iterations\n", p.show_every)

        end

        if p.Stest > 0 && p.test_every > 0
        
            @printf("Reporting test elbo every %d iterations\n", p.test_every)
        
        end

    end


    if p.Stest > 0 && p.test_every > 0 && mod(p.countiterations, p.test_every) == 0 
       
        currelbotest, currelbotest_std, numsamples = p.testelbofunction(p.bestμ, p.bestC)
       
        print(Crayon(foreground = :white, bold=false), @sprintf("Iteration %4d:\t elbo = %4.4f ± %4.4f \t test elbo (%4d) = ", p.countiterations, p.bestelbo, p.bestelbo_std, numsamples))
       
        
        if isempty(p.testelbohistory) || ~overfittingcriterion(currelbotest, currelbotest_std, p.bestelbo, p.bestelbo_std)#abs(currelbotest - p.bestelbo) < 3*(p.bestelbo_std + currelbotest_std)

            print(Crayon(foreground = :white, bold=false),  @sprintf("%4.4f ± %4.4f\n", currelbotest, currelbotest_std), Crayon(reset = true))

        else
            
            print(Crayon(foreground = :red, bold=true), @sprintf("%4.4f ± %4.4f\n", currelbotest, currelbotest_std), Crayon(reset = true))

            return true
        end

        push!(p.testelbohistory,     currelbotest)
        push!(p.testelbohistory_std, currelbotest_std)
        # plot(p)


    else
    
        if p.show_every > 0 && mod(p.countiterations, p.show_every) == 0
           
            print(Crayon(foreground = :white, bold=false), @sprintf("Iteration %4d:\t elbo = %4.4f ± %4.4f\n", p.countiterations, p.bestelbo, p.bestelbo_std), Crayon(reset = true))
        
        end

    end
    
    false

end


function overfittingcriterion(μtrain, σtrain, μtest, σtest)

    μdiff = μtrain - μtest

    σdiff = sqrt(σtrain^2 + σtest^2)

    # check below if value 0 is included in interval [μdiff - 3*σdiff, μdiff + 3*σdiff]

    if (μdiff - 3*σdiff < 0.0) && (0.0 < μdiff + 3*σdiff)
        
        return false # no overfitting detected
        
    end
    

    return true     # overfitting was detected

end