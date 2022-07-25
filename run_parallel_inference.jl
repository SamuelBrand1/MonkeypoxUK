using Distributed
addprocs(4)

@everywhere begin
    using Pkg
    Pkg.activate(".")
end

@everywhere begin
    using Distributions, StatsBase
    using LinearAlgebra, RecursiveArrayTools
    using OrdinaryDiffEq, ApproxBayes
    using JLD2, MLUtils
end

@everywhere include("mpxv_datawrangling.jl");
@everywhere include("uk_model_setup.jl");
@everywhere prior_vect = [Gamma(1, 1), Beta(5, 5), Gamma(3, 7 / 3), Beta(10, 90), LogNormal(0, 0.5), Gamma(2, 10 / 2),LogNormal(0,0.5)]
@everywhere setup = ABCSMC(mpx_sim_function, #simulation function
                        7, # number of parameters
                        0.5, #target ϵ
                        Prior(prior_vect); #Prior for each of the parameters
                        ϵ1=100,
                        convergence=0.05,
                        nparticles = 1000,
                        kernel=gaussiankernel,
                        constants=constants,
                        maxiterations=10^10)

