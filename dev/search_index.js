var documenterSearchIndex = {"docs":
[{"location":"technicaldescription/#Technical-background","page":"Technical description","title":"Technical background","text":"","category":"section"},{"location":"technicaldescription/","page":"Technical description","title":"Technical description","text":"We provide details regarding the implemented algorithm. In brief, the algorithm maximises the variational lower bound, typically known as the ELBO, using the reparametrisation trick.","category":"page"},{"location":"technicaldescription/#ELBO-maximisation","page":"Technical description","title":"ELBO maximisation","text":"","category":"section"},{"location":"technicaldescription/","page":"Technical description","title":"Technical description","text":"Our goal is to approximate the true (unnormalised) posterior  distribution p(thetamathcalD) with a Gaussian q(theta) = mathcalN(thetamuSigma). To achieve this, we maximising the expected lower bound:","category":"page"},{"location":"technicaldescription/","page":"Technical description","title":"Technical description","text":"mathcalL(muSigma) = int q(theta) log p(mathcalD theta) dtheta + mathcalHq,","category":"page"},{"location":"technicaldescription/","page":"Technical description","title":"Technical description","text":"also known as the ELBO. The above integral is a lower bound to the marginal likelihood int p(mathcalDtheta) dtheta geq mathcalL(muSigma) and is in general intractable. We can make progress by approximating it with as a Monte carlo average over S number of samples theta_ssim q(theta):","category":"page"},{"location":"technicaldescription/","page":"Technical description","title":"Technical description","text":"mathcalL(muSigma) approx frac1S sum_s=1^S log p(mathcalD theta_s) + mathcalHq.","category":"page"},{"location":"technicaldescription/","page":"Technical description","title":"Technical description","text":"Due to the sampling, however, the variational parameters no longer appear in the above approximation. Nevertheless, it is possible to re-introduce them by rewriting the sampled parameters as:","category":"page"},{"location":"technicaldescription/","page":"Technical description","title":"Technical description","text":"theta_s = mu + C z_s,","category":"page"},{"location":"technicaldescription/","page":"Technical description","title":"Technical description","text":"where z_ssimmathcalN(0I) and C is a matrix root of Sigma, i.e. CC^T = Sigma. We refer collectively to all samples as Z = z_1     z_S . This is known as the reparametrisation trick. Now we are able to we re-introduce the variational parameters in the approximated ELBO:","category":"page"},{"location":"technicaldescription/","page":"Technical description","title":"Technical description","text":"mathcalL_(FS)(muCZ) = frac1S sum_s=1^S log p(mathcalD mu + C z_s) + mathcalHq,","category":"page"},{"location":"technicaldescription/","page":"Technical description","title":"Technical description","text":"where the subscript FS stands for finite sample. We denote the approximate ELBO with mathcalL_(FS)(muCZ) and make it explicit that it depends on the samples Z.  By maximising the approximate ELBO mathcalL_(FS)(muCZ) with respect to the variational parameters mu and C we obtain the approximate posterior q(theta) = mathcalN(thetamuCC^T) that is the best Gaussian approximation to true posterior p(thetamathcalD).","category":"page"},{"location":"technicaldescription/#Choosing-the-number-of-samples-S","page":"Technical description","title":"Choosing the number of samples S","text":"","category":"section"},{"location":"technicaldescription/","page":"Technical description","title":"Technical description","text":"For large values of S the proposed approximate lower bound mathcalL_(FS)(muCZ) approximates the true bound mathcalL(muSigma) closely. Therefore, we expect that optimising mathcalL_(FS)(muCZ) will yield approximately the same variational parameters µ C as the optimisation of the intractable true lower bound mathcalL(muSigma) would.","category":"page"},{"location":"technicaldescription/","page":"Technical description","title":"Technical description","text":"Here the samples z_s are drawn at start of the algorithm and are kept fixed throughout its execution. The proposed scheme exhibits some fluctuation as mathcalL_(FS)(muCZ) depends on the random set of samples z_s that happened to be drawn at the start of the algorithm. Hence, for another set of randomly drawn samples z_s the function  mathcalL_(FS)(muCZ) will be (hopefully only slightly) different. However, for large enough S the fluctuation due to z_s should be innocuous and optimising it should yield approximately the same variational parameters for any drawn z_s.","category":"page"},{"location":"technicaldescription/","page":"Technical description","title":"Technical description","text":"However, if on the other hand we choose a small value for S, then the variational parameters will overly depend on the small set of samples z_s that happened to be drawn at the beginning of the algorithm. As a consequence, mathcalL_(FS)(muCZ) will approximate mathcalL(muSigma) poorly, and the resulting posterior q(theta) will also be a poor approximation to the true posterior. p(thetamathcalD). Hence, the variational parameters will be overadapted to the small set of samples z_s that happened to be drawn. Naturally, the question arises of how to choose a large enough S in order avoid the sitatuation where mathcalL_(FS)(muCZ) over-adapts the  variational parameters to the samples z_s. ","category":"page"},{"location":"technicaldescription/#Monitoring-ELBO-on-independent-test-set","page":"Technical description","title":"Monitoring ELBO on independent test set","text":"","category":"section"},{"location":"technicaldescription/","page":"Technical description","title":"Technical description","text":"A practical answer to diagnosing whether a sufficiently high number of samples S has been chosen, is the following: at the beginning of the algorithm we draw a second independent set of samples Z^prime = z_1^prime z_2^prime dots z_S^prime where S^prime is preferably a number larger than S. ","category":"page"},{"location":"technicaldescription/","page":"Technical description","title":"Technical description","text":"At each (or every few) iteration(s) we monitor the quantity mathcalL_(FS)(muCZ^prime) on the independent sample set Z^prime. If the variational parameters are not overadapting to the Z, then we should see that as the lower bound mathcalL_(FS)(muCZ)  increases, the quantity mathcalL_(FS)(muCZ^prime)  should also display a tendency to increase. If on the other hand the variational parameters are overadapting to  Z, then though mathcalL_(FS)(muCZ) is increasing, we will notice that mathcalL_(FS)(muCZ^prime)  is actually deteriorating. This is a clear sign that a larger S is required.","category":"page"},{"location":"technicaldescription/","page":"Technical description","title":"Technical description","text":"The described procedure is reminiscent of monitoring the generalisation performance of a learning algorithm on a validation set during training. A significant difference, however, is that while validation sets are typically of limited size, here we can set S^prime arbitrarily large. In practice, one may experiment with such values as e.g. S^prime = 2S or  S^prime = 10S. We emphasise that the samples in Z^prime are not used to optimise ELBO.","category":"page"},{"location":"technicaldescription/#Relevant-options-in-VI","page":"Technical description","title":"Relevant options in VI","text":"","category":"section"},{"location":"technicaldescription/","page":"Technical description","title":"Technical description","text":"This package implements variational inference using the re-parametrisation trick. Contrary to other flavours of this method, that repeatedly draw S new samples z_s at each iteration of the optimiser, here we draw at the start  a large number S of samples z_s and keep them fixed throughout the execution of the algorithm[1]. This avoids the difficulty of working with a noisy gradient and allows the use of optimisers like LBFGS. Using LBFGS, does away with the typical requirement of tuning learning rates (step sizes). However, this comes at the expense of risking overfitting to the samples z_s that happened to be drawn at the start. Because of fixing the samples  z_s, the algorithm doesn't not enjoy the same scalability as variational inference with stochastic gradient does. As a consequence,  the present package is recommented for problems with relatively few parameters, e.g. 2-20 parameters perhaps.","category":"page"},{"location":"technicaldescription/","page":"Technical description","title":"Technical description","text":"As explained in the previous section, one may monitor the approximate ELBO on an independent set of samples Z^prime of size S^prime. The package provides a mechanism for monitoring potential overfitting[2] via the options Stest and test_every. Options Stest set the number of test samples S^prime and test_every specifies how often we should monitor the approximate ELBO on Z^prime by evaluating mathcalL_(FS)(muCZ^prime). Please consult Evaluating the ELBO on test samples and the example Monitoring ELBO.","category":"page"},{"location":"technicaldescription/","page":"Technical description","title":"Technical description","text":"note: Note\nWhenever option Stest is set, test_every must be set to.","category":"page"},{"location":"technicaldescription/#Literature","page":"Technical description","title":"Literature","text":"","category":"section"},{"location":"technicaldescription/","page":"Technical description","title":"Technical description","text":"The work was independently developed and published here, (Arxiv link). Of course, the method has been widely popularised by the works Doubly Stochastic Variational Bayes for non-Conjugate Inference and Auto-Encoding Variational Bayes. The method seems to have appeared earlier in Fixed-Form Variational Posterior Approximation through Stochastic Linear Regression and again later in A comparison of variational approximations for fast inference in mixed logit models and perhaps in other publications too...","category":"page"},{"location":"technicaldescription/","page":"Technical description","title":"Technical description","text":"[1]: See paper, Algorithm 1.","category":"page"},{"location":"technicaldescription/","page":"Technical description","title":"Technical description","text":"[2]: See paper, Section 2.3.","category":"page"},{"location":"examples/#Infer-posterior-of-GP-hyperparameters","page":"Examples","title":"Infer posterior of GP hyperparameters","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"In the following we approximate the intractable posterior of the hyperparameters of a Gaussian process. In order to reproduce this example, certain packages need to be independently installed.","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"using ApproximateVI, Printf\nusing AbstractGPs, PyPlot, LinearAlgebra # These packages need to be independently installed\n\n\n# Define log-posterior assuming a flat prior over hyperparameters\n\nfunction logp(θ; x=x, y=y)\n\n    local kernel = exp(θ[2]) * (Matern52Kernel() ∘ ScaleTransform(exp(θ[1])))\n\n    local f = GP(kernel)\n\n    local fx = f(x, exp(θ[3]))\n\n    logpdf(fx, y)\n\nend\n\n\n# Generate some synthetic data\n\nN = 25  # number of data items\n\nσ = 0.2 # standard deviation of Gaussian noise\n\nx = rand(N)*10  # ranomly sample 1-dimensional inputs\n\ny = sin.(x) .+ randn(N)*σ # produce noise-corrupted outputs\n\nxtest = collect(-1.0:0.1:11.0) # test inputs\n\nytest = sin.(xtest) # test outputs\n\n\n\n# Approximate posterior with Gaussian\n\nq, = VI(θ -> logp(θ; x=x, y=y), randn(3)*2, S = 300, iterations = 1000, show_every = 10)\n\n\n# Draw samples from posterior and plot\n\nfor i in 1:3\n\n    # draw hyperparameter sample from approximating Gaussian distribution\n    local θ = rand(q)\n\n    # instantiate kernel\n    local sample_kernel = exp(θ[2]) * (Matern52Kernel() ∘ ScaleTransform(exp(θ[1])))\n\n    # intantiate kernel, GP object and calculate posterior mean and covariance for the training data x, y generated above\n    local f = GP(sample_kernel)\n    local p_fx = AbstractGPs.posterior(f(x, exp(θ[3])), y)\n    local μ, Σ = AbstractGPs.mean_and_cov(p_fx, xtest)\n\n    figure()\n    plot(x, y, \"ko\",label=\"Training data\")\n    plot(xtest, ytest, \"b-\", label=\"True curve\")\n\n    plot(xtest, μ, \"r-\")\n    fill_between(xtest, μ.-sqrt.(diag(Σ)),μ.+sqrt.(diag(Σ)), color=\"r\", alpha=0.3)\n\n    title(@sprintf(\"GP posterior, sampled hyperparameters %.2f, %.2f, %.2f\", exp(θ[1]),exp(θ[2]),exp(θ[3])))\n    legend()\n\nend","category":"page"},{"location":"examples/#Monitoring-ELBO","page":"Examples","title":"Monitoring ELBO","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"We use again as our target distribution an unnormalised Gaussian.","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"using ApproximateVI\n\n# instantiate a covariance matrix\nA = 0.1*randn(30, 30); Σ = A*A'\n\nlogp(x) = -sum(x'*(Σ\\x)) / 2\n\n# initial point\nx₀ = randn(30)","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"We set S=100 which is a quite low number of samples for inferring a 30-dimensional posterior. To diagnose whether S is set sufficiently high, we also test the ELBO on an indepedent set of samples of size Stest=3000 every test_every=20 iterations:","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"qlow, = VI(logp, x₀, S = 100, Stest = 3000, test_every = 10, iterations = 1000, gradientmode = :forward)","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"During execution we should see that there is a considerable gap between the ELBO and the test ELBO and that the test ELBO does not improve much. We now set S=1000:","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"qhigh, = VI(logp, x₀, S = 1000, Stest = 3000, test_every = 10, iterations = 1000, gradientmode = :forward)","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"In this case, we should see during execution that the reported ELBO is much closer to the test ELBO and the latter shows improvement. The improvement should also be visible in that the covariance of qhigh is closer to the true covariance:","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"norm(cov(qlow)  - Σ)    # this should be higher than value below\nnorm(cov(qhigh) - Σ)    # this should be lower  than value above","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"note: Note\nThe results of this example depend strongly on the covariance Sigma that we happened to sample.","category":"page"},{"location":"moreoptions/#Specifying-gradient-options","page":"More options","title":"Specifying gradient options","text":"","category":"section"},{"location":"moreoptions/","page":"More options","title":"More options","text":"Function VI allows the user to obtain a Gaussian approximation with minimal requirements. The user only needs to code a function logp that implements the log-posterior, provide an initial starting point x₀ and call:","category":"page"},{"location":"moreoptions/","page":"More options","title":"More options","text":"# log-posterior is a Gaussian with zero mean and unit covariance.\n# Hence, our approximation should be exact in this example.\nlogp(x) = -sum(x.*x) / 2\n\n# initial point implicitly specifies that the log-posterior is 5-dimensional\nx₀ = randn(5)\n\n# obtain approximation\nq, logev = VI(logp, x₀, S = 200, iterations = 10_000, show_every = 200)\n\n# Check that mean is close to zero and covariance close to identity.\n# mean and cov are re-exported function from Distributions.jl\nmean(q)\ncov(q)","category":"page"},{"location":"moreoptions/","page":"More options","title":"More options","text":"However, providing a gradient for logp can speed up the computation in VI.","category":"page"},{"location":"moreoptions/#Gradient-free-mode","page":"More options","title":"➤  Gradient free mode","text":"","category":"section"},{"location":"moreoptions/","page":"More options","title":"More options","text":"Specify by gradientmode = :gradientfree.","category":"page"},{"location":"moreoptions/","page":"More options","title":"More options","text":"If no options relating to the gradient are specified, i.e. none of the options gradientmode or gradlogp is specified, VI will by default use internally the Optim.NelderMead optimiser that does not need a gradient.  ","category":"page"},{"location":"moreoptions/","page":"More options","title":"More options","text":"The user can explicitly specify that VI should use the gradient free optimisation algorithm  Optim.NelderMead by setting gradientmode = :gradientfree.","category":"page"},{"location":"moreoptions/#Automatic-differentiation-mode","page":"More options","title":"➤  Automatic differentiation mode","text":"","category":"section"},{"location":"moreoptions/","page":"More options","title":"More options","text":"Specify by gradientmode = :forward.","category":"page"},{"location":"moreoptions/","page":"More options","title":"More options","text":"If logp is coding a differentiable function[1], then its gradient can be conveniently computed using automatic differentiation. By specifying gradientmode = :forward, function VI will internally use ForwardDiff to calculate the gradient of logp. In this case, VI will use internally the Optim.LBFGS optimiser.","category":"page"},{"location":"moreoptions/","page":"More options","title":"More options","text":"q, logev = VI(logp, x₀, S = 200, iterations = 30, show_every = 1, gradientmode = :forward)","category":"page"},{"location":"moreoptions/","page":"More options","title":"More options","text":"We note that with the use of gradientmode = :forward we arrive in fewer iterations to a result than in the gradient free case.","category":"page"},{"location":"moreoptions/#Gradient-provided","page":"More options","title":"➤  Gradient provided","text":"","category":"section"},{"location":"moreoptions/","page":"More options","title":"More options","text":"Specify by gradientmode = :provided.","category":"page"},{"location":"moreoptions/","page":"More options","title":"More options","text":"The user can provide a gradient for logp via the gradlogp option:","category":"page"},{"location":"moreoptions/","page":"More options","title":"More options","text":"# Let us calculate the gradient explicitly\ngradlogp(x) = -x\n\nq, logev = VI(logp, x₀, gradlogp = gradlogp, S = 200, iterations = 30, show_every = 1, gradientmode = :provided)","category":"page"},{"location":"moreoptions/","page":"More options","title":"More options","text":"In this case, VI will use internally the Optim.LBFGS optimiser. Again in this case we arrive in fewer iterations to a result than in the gradient free case.","category":"page"},{"location":"moreoptions/","page":"More options","title":"More options","text":"note: Note\nEven if a gradient has been explicitly provided via the gradlogp option, the user still needs to specify gradientmode = :provided to instruct VI to use the provided gradient.","category":"page"},{"location":"moreoptions/#Evaluating-the-ELBO-on-test-samples","page":"More options","title":"Evaluating the ELBO on test samples","text":"","category":"section"},{"location":"moreoptions/","page":"More options","title":"More options","text":"The options S specifies the number of samples to use when approximating the expected lower bound (ELBO), see Technical background. The higher the value we use for S, the better the approximation to the ELBO will be, but at a higher computational cost. The lower the value we use for S, the faster the computation will be, but the approximation to the ELBO may be poorer. Hence, when setting S we need to take this trade-off into account.","category":"page"},{"location":"moreoptions/","page":"More options","title":"More options","text":"Function VI offers a mechanism that tests whether the value S is set to a sufficiently high value. This mechanism makes use of two options, namely Stest and test_every. Option Stest defines a number of test samples used exclusively for evaluating (not optimising!) and reporting the ELBO every test_every number of iterations, see ELBO maximisation.  If S is set sufficiently high, then we should see that as the ELBO increases, so does the ELBO on the test samples. If on the other hand, we notice that the ELBO on the test samples is decreasing, then this is a clear sign that a larger S is required.","category":"page"},{"location":"moreoptions/","page":"More options","title":"More options","text":"Monitoring the ELBO this way is an effective way of detecting whether S has been set sufficiently high.","category":"page"},{"location":"moreoptions/","page":"More options","title":"More options","text":"The following code snippet shows how to specify the options Stest and test_every:","category":"page"},{"location":"moreoptions/","page":"More options","title":"More options","text":"# Use 2000 test samples and report test ELBO every 20 iterations\nq, logev = VI(logp, x₀, S = 200, iterations = 1000, Stest = 2000, test_every = 20)","category":"page"},{"location":"moreoptions/","page":"More options","title":"More options","text":"If the test ELBO at the current iteration is small than in the previous iteration, it is printed out in red colour. An additional example can be found here Monitoring ELBO.","category":"page"},{"location":"moreoptions/","page":"More options","title":"More options","text":"note: Note\nWhenever option Stest is set, test_every must be set to.","category":"page"},{"location":"moreoptions/","page":"More options","title":"More options","text":"[1]: The implementation of the function needs to satisfy certain requirements, see here.","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"VI\nexampleproblem1","category":"page"},{"location":"reference/#ApproximateVI.VI","page":"Reference","title":"ApproximateVI.VI","text":"Basic use:\n\nq, logev = VI(logp, μ, σ²=0.1; S = 100, iterations = 1, show_every = -1)\n\nReturns approximate Gaussian posterior and log evidence.\n\nArguments\n\nA description of only the most basic arguments follows.\n\nlogp is a function that expresses the (unnormalised) log-posterior, i.e. joint log-likelihood.\nμ is the initial mean of the approximating Gaussian posterior.\nσ² specifies the initial covariance of the approximating Gaussian posterior as σ² * I . Default value is 0.1.\nS is the number of drawn samples that approximate the lower bound integral.\niterations specifies for how many iterations to run optimisation on the lower bound (elbo).\nshow_every: report progress every show_every number of iterations. If set to value smaller than 1, then no progress will be reported.\n\nOutputs\n\nq is the approximating posterior returned as a Distributions.MvNormal type\nlogev is the approximate log-evidence.\n\nExample\n\n# infer posterior of Bayesian linear regression, compare to exact result\njulia> using LinearAlgebra, Distributions\njulia> D = 4; X = randn(D, 1000); W = randn(D); β = 0.3; α = 1.0;\njulia> Y = vec(W'*X); Y += randn(size(Y))/sqrt(β);\njulia> Sn = inv(α*I + β*(X*X')) ; mn = β*Sn*X*Y; # exact posterior\njulia> posterior, logev = VI( w -> logpdf(MvNormal(vec(w'*X), sqrt(1/β)), Y) + logpdf(MvNormal(zeros(D),sqrt(1/α)), w), randn(D); S = 1_000, iterations = 15);\njulia> display([mean(posterior) mn])\njulia> display([cov(posterior)  Sn])\njulia> display(logev) # display negative log evidence\n\n\n\n\n\n","category":"function"},{"location":"reference/#ApproximateVI.exampleproblem1","page":"Reference","title":"ApproximateVI.exampleproblem1","text":"Synthetic two-dimensional problem\n\nExample\n\njulia> logp = exampleproblem1() # target distribution to approximate\njulia> q, logev = VI(logp, randn(2), S = 100, iterations = 10_000, show_every = 50)\njulia> using Plots # must be indepedently installed.\njulia> x = -3:0.02:3\njulia> contour(x, x, map(x -> exp(logp(collect(x))), Iterators.product(x, x))', fill=true, c=:blues, colorbar = false) # plot target\njulia> contour!(x, x, map(x -> pdf(q,(collect(x))), Iterators.product(x, x))', color=\"red\", alpha=0.3) # plot approximation q\n\n\n\n\n\n","category":"function"},{"location":"#What's-this-package-for?","page":"Introduction","title":"What's this package for?","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"This package implements a particular type of approximate Bayesian inference: it approximates a posterior distribution with a full-covariance Gaussian distribution[1].","category":"page"},{"location":"#Basic-use","page":"Introduction","title":"Basic use","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"Currently, the main function of interest this package exposes is VI. At the very minimum, the user needs to provide a function that codes the (unnormalised) log-posterior function.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"Let's consider the following toy example:","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"using ApproximateVI\n\nlogp = exampleproblem1() # target log-posterior to approximate\nx₀ = randn(2)            # random initial mean for approximating Gaussian\nq, logev = VI(logp, randn(2), S = 100, iterations = 10_000, show_every = 50)\n\n# Plot target posterior, not log-posterior!\nusing Plots # must be indepedently installed.\nx = -3:0.02:3\ncontour(x, x, map(x -> exp(logp(collect(x))), Iterators.product(x, x))', fill=true, c=:blues)\n\n# Plot Gaussian approximation on top using red colour\ncontour!(x, x, map(x -> pdf(q,(collect(x))), Iterators.product(x, x))', color=\"red\", alpha=0.2)","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"A plot similar to the one below should appear. The filled blue contours correspond to the distribution being approximated, here the exponentiated logp, and the red contours correspond to the produced Gaussian approximation q.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"(Image: exampleproblem1)","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"Options S above specifies the number of samples to use in order to approximate the ELBO (see Technical background), i.e. the objective that which maximised produces the best Gaussian approximation. The higher the value of S is set, the better the approximation of the ELBO, however, at a higher computational cost. The lower the value of S is set, the faster the method, but the poorer the approximation of the ELBO. Options iterations specifies the number of iterations that the internal optimiser is run when maximising the ELBO. Option show_every specifies how often the progress of the ELBO maximisation should be reported.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"[1]: Approximate Variational Inference Based on a Finite Sample of Gaussian Latent Variables, [Arxiv].","category":"page"}]
}
