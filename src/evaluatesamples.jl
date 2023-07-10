function evaluatesamples(f, Z, ::Val{:parallel})

    Transducers.foldxt(+, Map(f),  Z) / length(Z)

end

function evaluatesamples(f, Z, ::Val{:serial})

    mapreduce(f, +, Z) / length(Z)

end