## Idea is to have both fitness and SBM effects in sexual contact

using Distributions, StatsBase, StatsPlots
using LinearAlgebra, RecursiveArrayTools
using OrdinaryDiffEq, ApproxBayes
using JLD2, MLUtils

## Grab UK data

include("mpxv_datawrangling.jl");
include("setup_model.jl");

##
_p = [0.01, 0.5, 20, 0.2, 0.5,6,1.5,130,1.0]

err, pred = mpx_sim_function_chp(_p, constants, mpxv_wkly)

plt = plot(pred)
scatter!(plt,mpxv_wkly)
display(plt)
print(err)

## Priors - chg point model

# α_choose, p_detect, mean_inf_period, p_trans, R0_other, M, init_scale ,chp_t,trans_red
prior_vect_cng_pnt = [Gamma(1, 1), # α_choose 1
                Beta(5, 5), #p_detect  2
                Gamma(3, 6 / 3), #mean_inf_period - 1  3
                Beta(5, 45), #p_trans  4
                LogNormal(log(0.75), 0.25), #R0_other 5
                Gamma(3, 10 / 3),#  M 6
                LogNormal(0,1),#init_scale 7
                Uniform(152,ts[end]),# chp_t 8
                Beta(1.5,1.5)]#trans_red 9
## Prior predictive checking - simulation
draws = [rand.(prior_vect_cng_pnt) for i = 1:1000]
prior_sims = map(θ -> mpx_sim_function_chp(θ,constants,mpxv_wkly),draws)
##Prior predictive checking - simulation
prior_preds = [sim[2] for sim in prior_sims]
plt = plot(;ylabel = "Weekly cases",
            title = "Prior predictive checking")
for pred in prior_preds
    plot!(plt,wks,pred,lab = "", color = :grey, alpha = 0.3)
end
display(plt)
savefig(plt,"plots/prior_predictive_checking_plot.png")

## Model-based calibration of target tolerance
min_mbc_errs = map(n -> minimum(map(x -> mpx_sim_function_chp(draws[n],constants,prior_sims[n][2])[1],1:5)),1:1000)


##
err_hist = histogram(min_mbc_errs,norm = :pdf,nbins = 200,
            lab = "",
            title = "Sampled errors from simulations with exact parameters",
            xlabel = "Median L1 relative error",
            size = (700,400))
vline!(err_hist,[0.58],lab = "5th percentile (rel. err. = 0.58)",lw = 3)
display(err_hist)
savefig(err_hist,"plots/mbc_error_calibration_plt.png")
##Run inference

setup_cng_pnt = ABCSMC(mpx_sim_function_chp, #simulation function
    9, # number of parameters
    0.58, #target ϵ
    Prior(prior_vect_cng_pnt); #Prior for each of the parameters
    ϵ1=100,
    convergence=0.05,
    nparticles = 1000,
    α = 0.5,
    kernel=gaussiankernel,
    constants=constants,
    maxiterations=10^10)

smc_cng_pnt = runabc(setup_cng_pnt, mpxv_wkly, verbose=true, progress=true)#, parallel=true)

##
@save("draws2_nrw.jld2",smc_cng_pnt)


##posterior predictive checking - simulation
post_preds = [part.other for part in smc_cng_pnt.particles]
plt = plot(;ylabel = "Weekly cases",
            title = "Posterior predictive checking")
for pred in post_preds
    plot!(plt,wks,pred,lab = "", color = :grey, alpha = 0.3)
end
scatter!(plt,wks,mpxv_wkly,lab = "Data")
display(plt)




