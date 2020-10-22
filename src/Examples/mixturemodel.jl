#######################################################################
# Example: approximate density given by mixture model with a Gaussian #
#######################################################################

using PyPlot

# Define means for three-component Gaussian mixture model
# All components are implicitly equally weighted and have unit covariance
μ = [zeros(2), [2.5; 0.0], [-2.5; 0.0]]

# Define log-likelihood
logp(θ) = log(exp(-0.5*sum((μ[1].-θ).^2)) + exp(-0.5*sum((μ[1].-θ).^2)) + exp(-0.5*sum((μ[3].-θ).^2)))

# Plot target density
θval = collect(LinRange(-5.0, 5.0, 100))
title("target unnormalised posterior")
contourf( repeat(θval,1,length(θval)),  repeat(θval',length(θval),1), map(θ -> exp(logp(θ)), [[x;y] for x in θval, y in θval]), cmap=plt.cm.binary)
axis("equal")

# Approximate mixture model with Gaussian density
posterior, logevidence = VI(logp, randn(2); S = 100, iterations = 30)

# Plot our approximation
plot_ellipse(posterior)
