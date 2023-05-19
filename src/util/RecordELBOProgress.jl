mutable struct RecordELBOProgress

    bestparam       # keep track of the parameters that lead to highest elbo on test latent Z

    show_every      # every how many iterations should we report the elbo

    test_every      # every how many iterations should we evaluate the elbo on test Z

    elbofunction

    testelbofunction

    bestsofarelbotest

    test_elbo_history

    test_elbo_history_std
    
end


function RecordELBOProgress(;initialparam = initialparam, show_every = show_every, test_every = test_every, testelbofunction = testelbofunction, elbo = elbo, unpack = unpack)

    RecordELBOProgress(initialparam, show_every, test_every, elbo, testelbofunction, -Inf, zeros(Float64, 0), zeros(Float64, 0))

end


# Get functions

getbestsolution(p::RecordELBOProgress) = p.bestparam

getbestelbo(p::RecordELBOProgress)     = p.bestsofarelbotest



function (p::RecordELBOProgress)(os) # used as callback
   
    iteration        =  os.iteration

    currentminimizer =  os.metadata["x"]

    currentelbo      =  - os.value

    # Unfortunately we need to re-evaluate the current best minimizer in order to get the std
    # This costs extra function evaluations!
   
    reevaluated_elbo, currentelbo_std = p.elbofunction(currentminimizer)

    # sanity check

    @assert(currentelbo == reevaluated_elbo)
 

    
    if iteration == 0
        
        if p.show_every > 0

            print(Crayon(foreground = :green, bold=false), @sprintf("Reporting elbo every %d iterations\n", p.show_every), Crayon(reset = true))
       
        end

        if p.test_every > 0
        
            print(Crayon(foreground = :green, bold=false), @sprintf("Reporting test elbo every %d iterations\n", p.test_every), Crayon(reset = true))
        
        end

    end


    if iteration > 0 && p.test_every > 0 && mod(iteration, p.test_every) == 0
       
        currelbotest, currelbotest_std, numsamples = p.testelbofunction(currentminimizer)

       
        print(Crayon(foreground = :white, bold=false), @sprintf("Iteration %4d:\t elbo = %4.4f ± %4.4f \t test elbo (n = %4d) = ", iteration, currentelbo, currentelbo_std, numsamples), Crayon(reset = true))
       
        
        if ~overfittingcriterion(currelbotest, currelbotest_std, currentelbo, currentelbo_std)
            
            print(Crayon(foreground = :white, bold=false),  @sprintf("%4.4f ± %4.4f\n", currelbotest, currelbotest_std), Crayon(reset = true))


            p.bestparam = copy(currentminimizer)

            p.bestsofarelbotest = (currelbotest, currelbotest_std)

            push!(p.test_elbo_history, currelbotest)

            push!(p.test_elbo_history_std, currelbotest_std)


        else
            
            print(Crayon(foreground = :red, bold=true), @sprintf("%4.4f ± %4.4f\n", currelbotest, currelbotest_std), Crayon(reset = true))

            return true

        end


    else
    
        if p.show_every > 0 && mod(iteration, p.show_every) == 0
           
            print(Crayon(foreground = :white, bold=false), @sprintf("Iteration %4d:\t elbo = %4.4f ± %4.4f\n", iteration, currentelbo, currentelbo_std), Crayon(reset = true))
        
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
    
    return true      # overfitting was detected

end



@recipe function f(p::RecordELBOProgress)
    
    x = 1:length(p.test_elbo_history)

    y = p.test_elbo_history

    seriestype --> :path

    x, y
    
end