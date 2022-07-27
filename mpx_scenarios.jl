using Distributions, StatsBase, StatsPlots
using LinearAlgebra, RecursiveArrayTools
using OrdinaryDiffEq, ApproxBayes
using JLD2, MLUtils

## Grab UK data and model set up

include("mpxv_datawrangling.jl");
include("setup_model.jl");

##Load posterior draws
filenames = ["draws1_nrw.jld2", "draws2_nrw.jld2"]
smcs = [load(filename)["smc_cng_pnt"] for filename in filenames];
particles = smcs[1].particles
particles = [particles; smcs[2].particles]
param_draws = [part.params for part in particles]

params_no_red = map(θ -> [θ[1:(end-1)]; 0.0], param_draws)

##Generate predictions with no change

long_wks = [wks; [wks[end] + Day(7 * k) for k = 1:12]]
long_mpxv_wkly = [mpxv_wkly; zeros(12)]

preds = map(θ -> mpx_sim_function_chp(θ, constants, long_mpxv_wkly)[2], param_draws)
preds_nored = map(θ -> mpx_sim_function_chp(θ, constants, long_mpxv_wkly)[2], params_no_red)
## Public health emergency effect forecast

wkly_vaccinations = [zeros(11); 1000; 2000; fill(5000, 11)]
chp_t2 = (Date(2022, 7, 23) - Date(2021, 12, 31)).value
trans_red2 = 0.5
inf_duration_red = 0.1
vac_effectiveness = 0.85
R0_other_red = 0

interventions = (; wkly_vaccinations, chp_t2, trans_red2, inf_duration_red, vac_effectiveness, R0_other_red)

# preds_interventions = map(θ -> mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, interventions)[2], param_draws)
# incidence_interventions = map(θ -> mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, interventions)[3], param_draws[1:2])
preds_and_incidence_interventions = map(θ -> mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, interventions)[2:3], param_draws)


##Simulation projections

median_pred = [median([preds[n][wk] for n = 1:length(preds)]) for wk in 1:length(long_wks)]
median_pred_no_red = [median([preds_nored[n][wk] for n = 1:length(preds_nored)]) for wk in 1:length(long_wks)]
median_pred_interventions = [median([preds_interventions[n][wk] for n = 1:length(preds_interventions)]) for wk in 1:length(long_wks)]

lb_pred_25 = median_pred .- [quantile([preds[n][wk] for n = 1:length(preds)], 0.25) for wk in 1:length(long_wks)]
lb_pred_025 = median_pred .- [quantile([preds[n][wk] for n = 1:length(preds)], 0.025) for wk in 1:length(long_wks)]
ub_pred_25 = [quantile([preds[n][wk] for n = 1:length(preds)], 0.75) for wk in 1:length(long_wks)] .- median_pred
ub_pred_025 = [quantile([preds[n][wk] for n = 1:length(preds)], 0.975) for wk in 1:length(long_wks)] .- median_pred

##
plt = plot(; ylabel="Weekly cases",
        title="UK MPVX reasonable worst case", #yscale=:log10,
        legend=:topleft,
        yticks=([1, 10, 100, 1000], [1, 10, 100, 1000]),
        left_margin=5mm,
        size=(800, 600), dpi=250)
plot!(plt, long_wks, median_pred, ribbon=(lb_pred_025, ub_pred_025), lw=0,
        color=:purple, fillalpha=0.3, lab="")
plot!(plt, long_wks, median_pred, ribbon=(lb_pred_25, ub_pred_25), lw=3,
        color=:purple, fillalpha=0.3, lab="Projection")
plot!(plt, long_wks, median_pred_no_red, lw=3,
        color=:red, fillalpha=0.3, lab="Projection (no red. in risk)")
plot!(plt, long_wks, median_pred_interventions, lw=3,
        color=:blue, fillalpha=0.3, lab="Projection (interventions)")
scatter!(plt, wks[[1, 2, end]], mpxv_wkly[[1, 2, end]], lab="Data (not used in fitting)", ms=6, color=:green)
scatter!(plt, wks[3:(end-1)], mpxv_wkly[3:(end-1)], lab="Data (used in fitting)", ms=6, color=:blue)

display(plt)
savefig(plt, "reasonable_worst_case.png")


