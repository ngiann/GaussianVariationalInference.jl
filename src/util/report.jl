function report(countiterations, show_every, test_every, Stest, elbo, bestμ, bestC, Ztest, bestelbo, currelbotest, prvelbotest)

    
    if countiterations == 1
        
        @printf("Reporting elbo every %d iterations\n", show_every)

        if Stest > 0 && test_every > 0
        
            @printf("Reporting test elbo every %d iterations\n", test_every)
        
        end

    end


    if test_every > 0 && mod(countiterations, test_every) == 0 && Stest > 0
       

        prvelbotest = currelbotest

        currelbotest = elbo(bestμ, bestC, Ztest)
       
        print(Crayon(foreground = :white, bold=false), @sprintf("Iteration %4d:\t elbo = %4.6f\t test elbo = ", countiterations, bestelbo))
       

        if currelbotest > prvelbotest

            print(Crayon(foreground = :white, bold=false),  @sprintf("%4.6f\n", currelbotest))

        else
            
            print(Crayon(foreground = :red, bold=true), @sprintf("%4.6f\n", currelbotest))

        end

    else
    
        if show_every > 0 && mod(countiterations, show_every) == 0
           
            print(Crayon(foreground = :white, bold=false), @sprintf("Iteration %4d:\t elbo = %4.6f\n", countiterations, bestelbo))
        
        end

    end
    
    currelbotest, prvelbotest

end