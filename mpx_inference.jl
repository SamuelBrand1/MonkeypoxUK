## Idea is to have both fitness and SBM effects in sexual contact

using Distributions, StatsBase, StatsPlots
using LinearAlgebra, RecursiveArrayTools
using OrdinaryDiffEq, ApproxBayes
using JLD2, MCMCChains, Roots
##import MonkeypoxUK
include("./src/dynamics.jl")
include("./src/inference.jl")

## Grab UK data
include("mpxv_datawrangling.jl");
include("setup_model.jl");

## Define priors for the parameters

prior_vect_cng_pnt = [Gamma(1, 1), # α_choose 1
    Beta(5, 5), #p_detect  2
    truncated(Gamma(3, 6 / 3), 0, 21), #mean_inf_period - 1  3
    Beta(1, 9), #p_trans  4
    LogNormal(log(0.75), 0.25), #R0_other 5
    Gamma(3, 10 / 3),#  M 6
    LogNormal(0, 1),#init_scale 7
    Uniform(135, 199),# chp_t 8
    Beta(1.5, 1.5),#trans_red 9
    Beta(1.5, 1.5),#trans_red_other 10
    Beta(1, 4),#trans_red WHO  11
    Beta(1, 4)]#trans_red_other WHO 12


## Use SBC for defining the ABC error target and generate prior predictive plots
##ϵ_target, plt_prc, hist_err = MonkeypoxUK.simulation_based_calibration(prior_vect_cng_pnt, wks, mpxv_wkly, constants)
ϵ_target, plt_prc, hist_err = simulation_based_calibration(prior_vect_cng_pnt, wks, mpxv_wkly, constants)

##setup_cng_pnt = ABCSMC(MonkeypoxUK.mpx_sim_function_chp, #simulation function
setup_cng_pnt = ABCSMC(mpx_sim_function_chp, #simulation function
    12, # number of parameters
    ϵ_target, #target ϵ derived from simulation based calibration
    Prior(prior_vect_cng_pnt); #Prior for each of the parameters
    ϵ1=1000,
    convergence=0.05,
    nparticles=2000,
    α=0.5,
    kernel=gaussiankernel,
    constants=constants,
    maxiterations=10^10)

##Run ABC    
smc_cng_pnt = runabc(setup_cng_pnt, mpxv_wkly, verbose=true, progress=true)#, parallel=true)
@save("posteriors/smc_posterior_draws_"*string(wks[end])*".jld2", smc_cng_pnt)
param_draws = [particle.params for particle in smc_cng_pnt.particles]
@save("posteriors/posterior_param_draws_"*string(wks[end])*".jld2", param_draws)

##posterior predictive checking - simple plot to see coherence of model with data

##smc_cng_pnt = load("posteriors/smc_posterior_draws_2022-06-27.jld2")["smc_cng_pnt"]
post_preds = [part.other for part in smc_cng_pnt.particles]
plt = plot(; ylabel="Weekly cases",
    title="Posterior predictive checking")
for pred in post_preds
    plot!(plt, wks[1:9], pred, lab="", color=[1 2], alpha=0.3)
end
scatter!(plt, wks, mpxv_wkly, lab=["Data: (MSM)" "Data: (non-MSM)"],ylims = (0,800))
display(plt)




