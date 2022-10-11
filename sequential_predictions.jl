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

colname = "seqn_fit5"
wks = Date.(past_mpxv_data_inferred_latest.week[1:size(mpxv_wkly, 1)], DateFormat("dd/mm/yyyy"))

inferred_prop_na_msm = past_mpxv_data_inferred_latest[:, colname] |> x -> x[.~ismissing.(x)]
inferred_prop_na_msm_lwr = past_mpxv_data_inferred_latest[:, "lower_"*colname] |> x -> x[.~ismissing.(x)]
inferred_prop_na_msm_upr = past_mpxv_data_inferred_latest[:, "upper_"*colname] |> x -> x[.~ismissing.(x)]



mpxv_wkly =
    Matrix(past_mpxv_data_inferred_latest[1:size(inferred_prop_na_msm, 1), ["gbmsm", "nongbmsm"]]) .+
    Vector(past_mpxv_data_inferred_latest[1:size(inferred_prop_na_msm, 1), "na_gbmsm"]) .*
    hcat(inferred_prop_na_msm, 1.0 .- inferred_prop_na_msm)

mpxv_wkly =
    past_mpxv_data_inferred_latest[1:size(inferred_prop_na_msm, 1), ["gbmsm", "nongbmsm"]] .+
    past_mpxv_data_inferred_latest[1:size(inferred_prop_na_msm, 1), "na_gbmsm"] .*
    hcat(inferred_prop_na_msm, 1.0 .- inferred_prop_na_msm) |> Matrix

lwr_mpxv_wkly =
    past_mpxv_data_inferred_latest[1:size(inferred_prop_na_msm, 1), ["gbmsm", "nongbmsm"]] .+
    past_mpxv_data_inferred_latest[1:size(inferred_prop_na_msm, 1), "na_gbmsm"] .*
    hcat(inferred_prop_na_msm_lwr, 1.0 .- inferred_prop_na_msm_lwr) |> Matrix


upr_mpxv_wkly =
    past_mpxv_data_inferred_latest[1:size(inferred_prop_na_msm, 1), ["gbmsm", "nongbmsm"]] .+
    past_mpxv_data_inferred_latest[1:size(inferred_prop_na_msm, 1), "na_gbmsm"] .*
    hcat(inferred_prop_na_msm_upr, 1.0 .- inferred_prop_na_msm_upr) |> Matrix


## Generate an ensemble of forecasts

seq_wks = [wks[1:4],wks[1:8], wks[1:12], wks[1:16], wks[1:20],wks]
seq_mpxv_wklys = [mpxv_wkly[1:4, :],mpxv_wkly[1:8, :], mpxv_wkly[1:12, :], mpxv_wkly[1:16, :], mpxv_wkly[1:20,:],mpxv_wkly]

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
function add_seqn_forecast!(plt, n; msm::Bool, N=5)
    period = (length(seq_wks[n])):(length(seq_wks[n])+11)
    k = msm ? 1 : 2
    plot!(plt,long_wks[period],seq_creds[n].mean_pred[period, k], color=get(ColorSchemes.cool,n/N),
    ribbon=(seq_creds[n].lb_pred_25[period, k], seq_creds[n].ub_pred_25[period, k]),
    fillalpha = 0.3, legend = :topleft,lab = seq_wks[n][end],lw = 0)

    plot!(plt,long_wks[period],seq_creds[n].mean_pred[period, k], color=get(ColorSchemes.cool,n/N),
    lab = "",lw = 3)
end


##

seq_proj_msm = plot(; ylabel="Weekly cases",
        title="UK Monkeypox Sequential Projections (GBMSM)",# yscale=:log10,
        legend=:topleft,
        # yticks=([1, 2, 11, 101, 1001], [0, 1, 10, 100, 1000]),
        # ylims=(0.8, 3001),
        xticks=([Date(2022, 5, 1) + Month(k) for k = 0:7], [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:7]),
        left_margin=5mm,right_margin=5mm,
        size=(800, 600), dpi=250,
        tickfont=18, titlefont=20, guidefont=24, legendfont=12)


for n = 1:5
    add_seqn_forecast!(seq_proj_msm,n;msm = true)
end
scatter!(seq_proj_msm,wks[1:(end)],mpxv_wkly[1:(end), 1],
        lab="Data available (6th Oct 2022)",
        ms=6, 
        color=:black,
        legend = :topright,
        yerrors = (mpxv_wkly[:, 1] .- lwr_mpxv_wkly[:, 1] , upr_mpxv_wkly[:, 1] .- mpxv_wkly[:, 1] ),
        )
display(seq_proj_msm)

##

seq_proj_nmsm = plot(; ylabel="Weekly cases",
        title="UK Monkeypox Sequential Projections (non-GBMSM)",# yscale=:log10,
        legend=:topleft,
        # yticks=([1, 2, 11, 101, 1001], [0, 1, 10, 100, 1000]),
        ylims=(-5, 200),
        xticks=([Date(2022, 5, 1) + Month(k) for k = 0:7], [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:7]),
        left_margin=5mm,right_margin=5mm,
        size=(800, 600), dpi=250,
        tickfont=18, titlefont=18, guidefont=24, legendfont=12)

##

for n = 1:5
    add_seqn_forecast!(seq_proj_nmsm,n;msm = false)
end
scatter!(seq_proj_nmsm,wks[1:(end)],mpxv_wkly[1:(end), 2],
        lab="Data available (6th Oct 2022)",
        ms=6,
        color=:black,
        legend = :topright,
        yerrors = (mpxv_wkly[:, 2] .- lwr_mpxv_wkly[:, 2] , upr_mpxv_wkly[:, 2] .- mpxv_wkly[:, 2] ),
        )
display(seq_proj_nmsm)


##        

savefig(seq_proj_msm,"plots/msm_sequential_forecasts.png")
savefig(seq_proj_nmsm,"plots/nmsm_sequential_forecasts.png")

##

layout = @layout [a b]
fig_seqn_proj = plot(
    seq_proj_msm,
    seq_proj_nmsm,
    size=(1750, 800),
    dpi=250,
    left_margin=10mm,
    bottom_margin=10mm,
    right_margin=10mm,
    top_margin=5mm,
    layout=layout,
)
savefig(fig_seqn_proj, "plots/seqn_forecasts.png")
