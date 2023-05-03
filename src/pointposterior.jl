function pointposterior()

    d1 = Normal(3,1)
    d2 = Normal(3,1.1)

    x = rand(d1)
    y = rand(d2)

    trange = -3:0.1:10
    
    post1(x) = pdf(d1,x)/(pdf(d1,x)+pdf(d2,x))
    
    p = map(post1, trange)

    @show mean(p)

    PyPlot.plot(trange, p)
end