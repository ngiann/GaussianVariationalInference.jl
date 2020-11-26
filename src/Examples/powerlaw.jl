###########################################################################
# Example: approximate posterior of power law described by two parameters #
###########################################################################

# We need:
# PyPlot to plot the posterior approximation
# Random to fix the random seed

using PyPlot, Random


# ensure repeatability

seed = 15

rg = MersenneTwister(seed)


# Define model and parameters responsible for generating the data below

trueparameter = [0.5; 0.5]

decayfunction(x,p) = p[1] .+ exp.(-p[2]*x)


# Generate some data

N = 20

σ = 0.05

x = rand(rg, N)*10.0

y = decayfunction(x, trueparameter) .+ σ*randn(rg, N)


# define log-likelihood function

decaylogpdf(x, y, p) = -sum(abs2.(decayfunction(x, p).-y))/(2*σ*σ)


# Approximate posterior with Gaussian

posteriorfull, =  VI( p->decaylogpdf(x,y,p), randn(rg, 2), S = 100, iterations = 50, show_every=1)

posteriormvi,  = MVI( p->decaylogpdf(x,y,p), randn(rg, 2), S = 100, iterations = 50, show_every=1)


# Plot data

figure(1)

cla()

plot(x, y, "bo", label="observations")

xrange = collect(LinRange(minimum(x), maximum(x), 100))

plot(xrange, map(xi->decayfunction(xi, trueparameter), xrange), "k-", label="true power law")

legend()


# Plot true unnormalised posterior

figure(2)

cla()

prange = 0.3:0.001:0.9

contourf( repeat(prange,1,length(prange)),  repeat(prange',length(prange),1), map(p -> exp(decaylogpdf(x,y,p)), [[x;y] for x in prange, y in prange]), cmap=plt.cm.binary)

xlabel("p1")

ylabel("p2")


# Plot our Gaussian approximation

ApproximateVI.plot_ellipse(posteriorfull, "b", "full")

ApproximateVI.plot_ellipse(posteriormvi,  "g", "mvi")


# Add the true parameter on the plot

plot(trueparameter[1], trueparameter[2], "mo", label="true parameter")

legend()

nothing
