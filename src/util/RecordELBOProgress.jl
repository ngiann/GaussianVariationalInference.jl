mutable struct RecordELBOProgress

    bestelbo # default -Inf
    bestμ 
    bestC

    countiterations # default is 0

    Stest
    show_every
    test_every
    testelbofunction
    testelbohistory
    
end


function RecordELBOProgress(; μ = μ, C = C, Stest = Stest, show_every = show_every, test_every = test_every, elbo = elbo, seed = seed)
    
    D = length(μ)

    Ztest = generatelatentZ(S = Stest, D = D, seed = seed + 1)

    testeblofunction(μ, C) = elbo(μ, C, Ztest)

    RecordELBOProgress(-Inf, μ, C, 0, Stest, show_every, test_every, testeblofunction, [-Inf])

end


function update!(p::RecordELBOProgress; newelbo = newelbo, μ = μ, C = C)

    if newelbo > p.bestelbo
        p.bestelbo = newelbo
        p.bestμ .= μ
        p.bestC .= C
    end

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
       
        currelbotest = p.testelbofunction(p.bestμ, p.bestC)
       
        print(Crayon(foreground = :white, bold=false), @sprintf("Iteration %4d:\t elbo = %4.6f\t test elbo = ", p.countiterations, p.bestelbo))
       

        if currelbotest > p.testelbohistory[end]

            print(Crayon(foreground = :white, bold=false),  @sprintf("%4.6f\n", currelbotest))

        else
            
            print(Crayon(foreground = :red, bold=true), @sprintf("%4.6f\n", currelbotest))

        end

        push!(p.testelbohistory, currelbotest)

    else
    
        if p.show_every > 0 && mod(p.countiterations, p.show_every) == 0
           
            print(Crayon(foreground = :white, bold=false), @sprintf("Iteration %4d:\t elbo = %4.6f\n", p.countiterations, p.bestelbo))
        
        end

    end
    
    false

end
