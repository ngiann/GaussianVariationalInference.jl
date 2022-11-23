# ApproximateVI.jl


## What's this package for?

Approximate a target posterior with a Gaussian by minimising Kullback-Leibler divergence.


## Basic use

Currently, the package exposes a single function called `VI`.
At the very minimum, the user needs to provide a function that codes the (unnormalised)log-posterior function.

Let's consider the following toy example:





