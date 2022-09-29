using Distributions, StatsBase, StatsPlots
using LinearAlgebra, RecursiveArrayTools
using OrdinaryDiffEq, ApproxBayes
using JLD2, MCMCChains
using MonkeypoxUK
using ColorSchemes
## Grab UK data and model set up
include("mpxv_datawrangling_inff.jl");
include("setup_model.jl");

## Comment out to use latest data rather than reterospective data

colname = "seqn_fit4"
inferred_prop_na_msm = past_mpxv_data_inferred[:, colname] |> x -> x[.~ismissing.(x)]
mpxv_wkly =
    Matrix(past_mpxv_data_inferred[1:size(inferred_prop_na_msm, 1), ["gbmsm", "nongbmsm"]]) .+
    Vector(past_mpxv_data_inferred[1:size(inferred_prop_na_msm, 1), "na_gbmsm"]) .*
    hcat(inferred_prop_na_msm, 1.0 .- inferred_prop_na_msm)
wks = Date.(past_mpxv_data_inferred.week[1:size(mpxv_wkly, 1)], DateFormat("dd/mm/yyyy"))

## Generate an ensemble of forecasts

seq_wks = [wks[1:8], wks[1:12], wks[1:16], wks]
seq_mpxv_wklys = [mpxv_wkly[1:8, :], mpxv_wkly[1:12, :], mpxv_wkly[1:16, :], mpxv_wkly]

function load_smc(wks)
    wk = wks[end]
    load("posteriors/posterior_param_draws_" * string(wk) * ".jld2")["param_draws"]
end

seq_param_draws = map(load_smc, seq_wks)

seq_forecasts = map((param_draws, wks, mpxv_wkly) -> generate_forecast_projection(param_draws, wks, mpxv_wkly, constants),
    seq_param_draws,
    seq_wks,
    seq_mpxv_wklys)

##
preds = [[x[1] for x in forecast] for forecast in seq_forecasts]
seq_creds = MonkeypoxUK.cred_intervals.(preds)
long_wks = [wks; [wks[end] + Day(7 * k) for k = 1:12]]
long_mpxv_wkly = [mpxv_wkly; zeros(12, 2)]

##
"""
function add_seqn_forecast!(plt, n; msm::Bool, N=4)

Add the `n` the sequential prediction curve to the plot.    
"""
function add_seqn_forecast!(plt, n; msm::Bool, N=4)
    period = (length(seq_wks[n])):(length(seq_wks[n])+11)
    k = msm ? 1 : 2
    plot!(plt,long_wks[period],seq_creds[n].mean_pred[period, k], color=get(ColorSchemes.cool,n/N),
    ribbon=(seq_creds[n].lb_pred_10[period, k], seq_creds[n].ub_pred_10[period, k]),
    fillalpha = 0.3, legend = :topleft,lab = seq_wks[n][end],lw = 0)

    plot!(plt,long_wks[period],seq_creds[n].mean_pred[period, k], color=get(ColorSchemes.cool,n/N),
    lab = "",lw = 3)
end


##

seq_proj_msm = plot(; ylabel="Weekly cases",
        title="UK Monkeypox Sequential Projections (MSM)",# yscale=:log10,
        legend=:topleft,
        # yticks=([1, 2, 11, 101, 1001], [0, 1, 10, 100, 1000]),
        # ylims=(0.8, 3001),
        xticks=([Date(2022, 5, 1) + Month(k) for k = 0:6], [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:6]),
        left_margin=5mm,
        size=(800, 600), dpi=250,
        tickfont=11, titlefont=18, guidefont=18, legendfont=11)


for n = 1:4
    add_seqn_forecast!(seq_proj_msm,n;msm = true)
end
scatter!(seq_proj_msm,wks[1:(end)],mpxv_wkly[1:(end), 1],
        lab="Data",
        ms=6, color=:black)
display(seq_proj_msm)

##

##

seq_proj_nmsm = plot(; ylabel="Weekly cases",
        title="UK Monkeypox Sequential Projections (non-MSM)",# yscale=:log10,
        legend=:topleft,
        # yticks=([1, 2, 11, 101, 1001], [0, 1, 10, 100, 1000]),
        ylims=(-5, 200),
        xticks=([Date(2022, 5, 1) + Month(k) for k = 0:6], [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:6]),
        left_margin=5mm,
        size=(800, 600), dpi=250,
        tickfont=11, titlefont=18, guidefont=18, legendfont=11)

##

for n = 1:4
    add_seqn_forecast!(seq_proj_nmsm,n;msm = false)
end
scatter!(seq_proj_nmsm,wks[1:(end)],mpxv_wkly[1:(end), 2],
        lab="Data",
        ms=6, color=:black)
display(seq_proj_nmsm)


##        

##

savefig(seq_proj_msm,"plots/msm_sequential_forecasts.png")
savefig(seq_proj_nmsm,"plots/nmsm_sequential_forecasts.png")