function pickoptimiser(μ, logp, gradlogp, gradientmode)

    if gradientmode == :forward
        
        # optimiser to be used with gradient calculated with automatic differentiation

        return ConjugateGradient(), x -> ForwardDiff.gradient(logp, x)

    elseif gradientmode == :zygote
        
        # optimiser to be used with gradient calculated with automatic differentiation
    
        return ConjugateGradient(), x -> Zygote.gradient(logp, x)[1]

    elseif gradientmode == :reverse

        compiled_f_tape = ReverseDiff.compile(ReverseDiff.GradientTape(logp, μ))

        local storage = zeros(length(μ))

        function helper(p)
            
            ReverseDiff.gradient!(storage, compiled_f_tape, p)

            return storage

        end
   
        return ConjugateGradient(), helper

    elseif gradientmode == :provided

        if any(isnan.(gradlogp(μ)))
            
            error("provided gradient returns NaN when evaluate at provided μ")

        end

        # optimiser to be used with user provided gradient

        return ConjugateGradient(), gradlogp

    elseif gradientmode == :gradientfree
        
        return NelderMead(), gradlogp

    else

        error("invalid specification of argument gradientmode")

    end

end
