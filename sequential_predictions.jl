using Distributions, StatsBase, StatsPlots, Plots.PlotMeasures
using LinearAlgebra, RecursiveArrayTools, CSV
using OrdinaryDiffEq, ApproxBayes, DataFrames
using JLD2, MCMCChains, ProgressMeter
using MonkeypoxUK
using ColorSchemes, Dates

## MSM data with data inference

past_mpxv_data_inferred =
    CSV.File("data/weekly_data_imputation_2022-09-30.csv", missingstring = "NA") |>
    DataFrame

wks = Date.(past_mpxv_data_inferred.week, DateFormat("dd/mm/yyyy"))

include("setup_model.jl");

## Format case data into expected (over GBDT inference) case incidence

colname = "seqn_fit5"

inferred_prop_na_msm = past_mpxv_data_inferred[:, colname] |> x -> x[.~ismissing.(x)]
inferred_prop_na_msm_lwr =
    past_mpxv_data_inferred[:, "lower_"*colname] |> x -> x[.~ismissing.(x)]
inferred_prop_na_msm_upr =
    past_mpxv_data_inferred[:, "upper_"*colname] |> x -> x[.~ismissing.(x)]

mpxv_wkly =
    past_mpxv_data_inferred[1:size(inferred_prop_na_msm, 1), ["gbmsm", "nongbmsm"]] .+
    past_mpxv_data_inferred[1:size(inferred_prop_na_msm, 1), "na_gbmsm"] .*
    hcat(inferred_prop_na_msm, 1.0 .- inferred_prop_na_msm) |> Matrix

lwr_mpxv_wkly =
    past_mpxv_data_inferred[1:size(inferred_prop_na_msm, 1), ["gbmsm", "nongbmsm"]] .+
    past_mpxv_data_inferred[1:size(inferred_prop_na_msm, 1), "na_gbmsm"] .*
    hcat(inferred_prop_na_msm_lwr, 1.0 .- inferred_prop_na_msm_lwr) |> Matrix

upr_mpxv_wkly =
    past_mpxv_data_inferred[1:size(inferred_prop_na_msm, 1), ["gbmsm", "nongbmsm"]] .+
    past_mpxv_data_inferred[1:size(inferred_prop_na_msm, 1), "na_gbmsm"] .*
    hcat(inferred_prop_na_msm_upr, 1.0 .- inferred_prop_na_msm_upr) |> Matrix



##

wks = Date.(past_mpxv_data_inferred.week[1:size(mpxv_wkly, 1)], DateFormat("dd/mm/yyyy"))
ts = wks .|> d -> d - Date(2021, 12, 31) .|> t -> t.value

wkly_vaccinations = [
    [zeros(12); 1000; 2000; fill(5000, 4)] * 1.675
    fill(650, 18)
]

## Generate an ensemble of forecasts

seq_wks = [wks[1:8], wks[1:12], wks[1:16], wks[1:20], wks]

seq_mpxv_wklys = [
    mpxv_wkly[1:8, :],
    mpxv_wkly[1:12, :],
    mpxv_wkly[1:16, :],
    mpxv_wkly[1:20, :],
    mpxv_wkly,
]

##

## Load posterior draws and saved trajectiories

description_str = "no_ngbmsm_chg" #<---- This is the main model
# description_str = "no_bv_cng" #<---- This is the version of the model with no behavioural change
# description_str = "one_metapop" #<--- This is the version of the model with no metapopulation structure
# description_str = "" #<--- this is the older version main model


function load_posteriors_for_projection(date_str, description_str; pheic_effect = true)
    param_draws =
    load("posteriors/posterior_param_draws_" * date_str * description_str * ".jld2")["param_draws"]
    detected_cases =
        load("posteriors/posterior_detected_cases_" * date_str * description_str * ".jld2")["detected_cases"]
    vac_effs = load(
        "posteriors/posterior_vac_effectivenesses_" * date_str * description_str * ".jld2",
    )["vac_effectivenesses"]
    end_states =
        load("posteriors/posterior_end_states_" * date_str * description_str * ".jld2")["end_states"]
    
    if !pheic_effect

    end

    return (; param_draws, detected_cases, vac_effs, end_states)
end



post_draws = load_posteriors_for_projection(string(seq_wks[2][end]), "no_ngbmsm_chg")

cred = cred_intervals(post_draws.detected_cases, central_est = :median)


##
n_samples = 2000

proj_fromend = [
    (
        ts = [seq_wks[1][end] + Week(k) for k = 1:12] .|> d -> (d - Date(2021, 12, 31)).value |> Float64,
        wkly_vaccinations = wkly_vaccinations[(length(seq_wks[1])+1):end],
        vac_effectiveness = post_draws.vac_effs[k],
    ) for k = 1:n_samples
]


##
projections_from_end = @showprogress 0.1 "MPX Forecasts:" map(
    (θ, interventions, state) ->
        mpx_sim_function_projections(θ, constants, interventions, state),
    post_draws.param_draws,
    proj_fromend,
    post_draws.end_states,
)

cred_proj = MonkeypoxUK.cred_intervals(
    [proj.detected_cases for proj in projections_from_end],
    central_est = :median,
)

d1 = size(cred_proj.median_pred, 1)

plot(seq_wks[1][3:end], cred.median_pred,
           ribbon = (cred.lb_pred_25, cred.ub_pred_25))
scatter!(wks, mpxv_wkly)           
plot!([seq_wks[1][end] + Week(k) for k = 1:12], cred_proj.median_pred,
        ribbon = (cred_proj.lb_pred_25, cred_proj.ub_pred_25))

## CHANGE BELOW
seq_param_draws = map(load_smc, seq_wks)

seq_forecasts = map(
    (param_draws, wks, mpxv_wkly) ->
        generate_forecast_projection(param_draws, wks, mpxv_wkly, constants),
    seq_param_draws,
    seq_wks,
    seq_mpxv_wklys,
)



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
function add_seqn_forecast!(plt, n; msm::Bool, N = 5)
    period = (length(seq_wks[n])):(length(seq_wks[n])+11)
    k = msm ? 1 : 2
    plot!(
        plt,
        long_wks[period],
        seq_creds[n].mean_pred[period, k],
        color = get(ColorSchemes.cool, n / N),
        ribbon = (seq_creds[n].lb_pred_10[period, k], seq_creds[n].ub_pred_10[period, k]),
        fillalpha = 0.3,
        legend = :topleft,
        lab = seq_wks[n][end],
        lw = 0,
    )

    plot!(
        plt,
        long_wks[period],
        seq_creds[n].mean_pred[period, k],
        color = get(ColorSchemes.cool, n / N),
        lab = "",
        lw = 3,
    )
end


##

seq_proj_msm = plot(;
    ylabel = "Weekly cases",
    title = "UK Monkeypox Sequential Projections (GBMSM)",# yscale=:log10,
    legend = :topleft,
    # yticks=([1, 2, 11, 101, 1001], [0, 1, 10, 100, 1000]),
    # ylims=(0.8, 3001),
    xticks = (
        [Date(2022, 5, 1) + Month(k) for k = 0:7],
        [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:7],
    ),
    left_margin = 5mm,
    right_margin = 5mm,
    size = (800, 600),
    dpi = 250,
    tickfont = 18,
    titlefont = 20,
    guidefont = 24,
    legendfont = 12,
)


for n = 1:5
    add_seqn_forecast!(seq_proj_msm, n; msm = true)
end
scatter!(
    seq_proj_msm,
    wks[1:(end)],
    mpxv_wkly[1:(end), 1],
    lab = "Data available (6th Oct 2022)",
    ms = 6,
    color = :black,
    legend = :topright,
    yerrors = (
        mpxv_wkly[:, 1] .- lwr_mpxv_wkly[:, 1],
        upr_mpxv_wkly[:, 1] .- mpxv_wkly[:, 1],
    ),
)
display(seq_proj_msm)

##

seq_proj_nmsm = plot(;
    ylabel = "Weekly cases",
    title = "UK Monkeypox Sequential Projections (non-GBMSM)",# yscale=:log10,
    legend = :topleft,
    # yticks=([1, 2, 11, 101, 1001], [0, 1, 10, 100, 1000]),
    ylims = (-5, 200),
    xticks = (
        [Date(2022, 5, 1) + Month(k) for k = 0:7],
        [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:7],
    ),
    left_margin = 5mm,
    right_margin = 5mm,
    size = (800, 600),
    dpi = 250,
    tickfont = 18,
    titlefont = 18,
    guidefont = 24,
    legendfont = 12,
)

##

for n = 1:5
    add_seqn_forecast!(seq_proj_nmsm, n; msm = false)
end
scatter!(
    seq_proj_nmsm,
    wks[1:(end)],
    mpxv_wkly[1:(end), 2],
    lab = "Data available (6th Oct 2022)",
    ms = 6,
    color = :black,
    legend = :topright,
    yerrors = (
        mpxv_wkly[:, 2] .- lwr_mpxv_wkly[:, 2],
        upr_mpxv_wkly[:, 2] .- mpxv_wkly[:, 2],
    ),
)
display(seq_proj_nmsm)


##        

savefig(seq_proj_msm, "plots/msm_sequential_forecasts.png")
savefig(seq_proj_nmsm, "plots/nmsm_sequential_forecasts.png")

##

layout = @layout [a b]
fig_seqn_proj = plot(
    seq_proj_msm,
    seq_proj_nmsm,
    size = (1750, 800),
    dpi = 250,
    left_margin = 10mm,
    bottom_margin = 10mm,
    right_margin = 10mm,
    top_margin = 5mm,
    layout = layout,
)
display(fig_seqn_proj)
savefig(fig_seqn_proj, "plots/seqn_forecasts.png")
