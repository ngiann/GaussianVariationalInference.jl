#######################################################################
# Example: approximate density given by mixture model with a Gaussian #
#######################################################################

using PyPlot

# Define means for three-component Gaussian mixture model
# All components are implicitly equally weighted and have unit covariance
μ = [zeros(2), [2.9; 0.0], [2.5; 1.2]]

# Define log-likelihood
logp(θ) = log(exp(-0.5*sum((μ[1].-θ).^2)) + exp(-0.5*sum((μ[2].-θ).^2)) + exp(-0.5*sum((μ[3].-θ).^2)))

# Plot target density
figure()
θval = collect(LinRange(-5.0, 5.0, 100))
title("target unnormalised posterior")
contourf( repeat(θval,1,length(θval)),  repeat(θval',length(θval),1), map(θ -> exp(logp(θ)), [[x;y] for x in θval, y in θval]), cmap=plt.cm.binary)
axis("equal")

# Approximate mixture model with Gaussian density
posteriorfull,   =         VI(logp, randn(2); S = 150, iterations = 100)
posteriordiag,   =     VIdiag(logp, randn(2); S = 150, iterations = 100)
posteriorfixC,   = VIfixedcov(logp, randn(2), 1.0*Matrix(I, 2, 2); S = 150, iterations = 100)
posteriorlap,    =    laplace(logp, randn(2); iterations = 100)
posteriormvi,    =        MVI(logp, posteriorlap; S = 150, iterations = 100)
# posteriormvidiag,  = MVI(logp, MvNormal(zeros(2),Matrix(I,2,2)); S = 150, iterations = 100)

# Plot our approximations
ApproximateVI.plot_ellipse(posteriorfull,   "b", "full")
ApproximateVI.plot_ellipse(posteriordiag,   "r", "diag")
ApproximateVI.plot_ellipse(posteriorfixC,   "c", "sphere-fix")
ApproximateVI.plot_ellipse(posteriormvi,    "g", "mvi")
ApproximateVI.plot_ellipse(posteriorlap,    "m", "laplace")
# ApproximateVI.plot_ellipse(posteriormvidiag,  "k", "diag by mvi")

legend();

@info("Note how the covariance matrices of Laplace and MVI are rotated the same way")
