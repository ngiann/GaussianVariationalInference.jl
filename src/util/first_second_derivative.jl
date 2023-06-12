firstderivative(f, x)  = ForwardDiff.derivative(f, x)

secondderivative(f, x) = ForwardDiff.derivative(firstderivativefunction(f), x)


firstderivativefunction(f)  = x -> ForwardDiff.derivative(f, x)

function secondderivativefunction(f) 
    
    aux = firstderivativefunction(f)
    
    firstderivativefunction(aux)

end
