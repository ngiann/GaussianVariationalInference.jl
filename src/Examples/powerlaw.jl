###########################################################################
# Example: approximate posterior of power law described by two parameters #
###########################################################################

# We need PyPlot to plot the posterior approximation

using PyPlot, Distributions, Random

# ensure repeatability
seed = 101

rg = MersenneTwister(seed)

# Define model and parameters responsible for generating the data below

trueparameter = [0.75;0.4]

decayfunction(x,p) = p[1] .+ exp.(-p[2]*x)


# Generate some data

N = 12

σ = 0.075

x = rand(rg, N)*7.0

y = decayfunction(x, trueparameter) .+ σ*randn(rg, N)


# define log-likelihood function

decaylogpdf(x, y, p) = logpdf(MvNormal(decayfunction(x, p), σ), y)


# Approximate posterior with Gaussian

posterior, = VI( p->decaylogpdf(x,y,p), [randn(rg, 2) for i=1:5], S = 100, iterations = 50, show_every=1)


# Plot data

figure(1)

cla()

plot(x, y, "bo", label="observations")

xrange = collect(LinRange(minimum(x), maximum(x), 100))

plot(xrange, map(xi->decayfunction(xi, trueparameter), xrange), "k-", label="true power law")


# Plot true unnormalised posterior

figure(2)

cla()

prange = 0.001:0.01:1.5

contourf( repeat(prange,1,length(prange)),  repeat(prange',length(prange),1), map(p -> exp(decaylogpdf(x,y,p)), [[x;y] for x in prange, y in prange]), cmap=plt.cm.binary)

xlabel("p1")

ylabel("p2")


# Plot our Gaussian approximation

ApproximateVI.plot_ellipse(posterior)


# Add the true parameter on the plot

plot(trueparameter[1], trueparameter[2], "ro")
