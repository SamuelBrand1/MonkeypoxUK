using Distributions, StatsBase, StatsPlots
using LinearAlgebra, RecursiveArrayTools
using OrdinaryDiffEq, ApproxBayes
using JLD2, MCMCChains
import MonkeypoxUK
## Grab UK data and setup model
include("mpxv_datawrangling.jl");
include("setup_model.jl");

## Generate an ensemble of forecasts

seq_wks = [wks[1:5], wks[1:9], wks[1:14]]
seq_mpxv_wklys = [mpxv_wkly[1:5, :], mpxv_wkly[1:9, :], mpxv_wkly[1:14, :]]

function load_smc(wks)
    wk = wks[end]
    load("posteriors/posterior_param_draws_" * string(wk) * ".jld2")["param_draws"]
end
seq_param_draws = map(load_smc, seq_wks)

seq_forecasts = map((param_draws, wks, mpxv_wkly) -> MonkeypoxUK.generate_forecast_projections(param_draws, wks, mpxv_wkly, constants),
    seq_param_draws,
    seq_wks,
    seq_mpxv_wklys)

##
seq_creds = MonkeypoxUK.cred_intervals.([f[1] for f in seq_forecasts])

plot(seq_creds[1].mean_pred[:, 1], color=1,
    ribbon=(seq_creds[1].lb_pred_25[:, 1], seq_creds[1].ub_pred_25[:, 1]),
    fillalpha = 0.3, legend = :topleft,lab = seq_wks[1][end])
plot!(seq_creds[2].mean_pred[:, 1], color=2,
    ribbon=(seq_creds[2].lb_pred_25[:, 1], seq_creds[2].ub_pred_25[:, 1]),
    fillalpha=0.3, lab=seq_wks[2][end])
plot(seq_creds[3].mean_pred[:, 1], color=3,
    ribbon=(seq_creds[3].lb_pred_025[:, 1], seq_creds[3].ub_pred_025[:, 1]),
    fillalpha=0.3, lab=seq_wks[3][end])
scatter!(mpxv_wkly[:,1])