using Distributions, StatsBase, StatsPlots
using Plots.PlotMeasures, CSV, DataFrames
using LinearAlgebra, RecursiveArrayTools
using OrdinaryDiffEq, ApproxBayes, MCMCChains
using JLD2
using MonkeypoxUK

## Grab UK data and model set up
include("mpxv_datawrangling_inff.jl");
include("setup_model.jl");

## Comment out to use latest data rather than reterospective data

colname = "seqn_fit5"
inferred_prop_na_msm = past_mpxv_data_inferred[:, colname] |> x -> x[.~ismissing.(x)]
mpxv_wkly =
    past_mpxv_data_inferred[1:size(inferred_prop_na_msm, 1), ["gbmsm", "nongbmsm"]] .+
    past_mpxv_data_inferred[1:size(inferred_prop_na_msm, 1), "na_gbmsm"] .*
    hcat(inferred_prop_na_msm, 1.0 .- inferred_prop_na_msm) |> Matrix
wks = Date.(past_mpxv_data_inferred.week[1:size(mpxv_wkly, 1)], DateFormat("dd/mm/yyyy"))
ts = wks .|> d -> d - Date(2021, 12, 31) .|> t -> t.value
# wkly_vaccinations = [zeros(12); 1000; 2000; fill(5000, 23)] * 1.55
wkly_vaccinations = [
    [zeros(12); 1000; 2000; fill(5000, 4)] * 1.675
    fill(650, 19)
]
print("cum. vacs = $(sum(wkly_vaccinations))")

##Load posterior draws
param_draws =
    load("posteriors/posterior_param_draws_2022-09-26_binom_bf.jld2")["param_draws"]

##

n_lookaheadweeks = 26
long_wks = [wks; [wks[end] + Day(7 * k) for k = 1:n_lookaheadweeks]]
long_mpxv_wkly = [mpxv_wkly; zeros(n_lookaheadweeks, 2)]
wkly_vaccinations = [wkly_vaccinations[1:26]; fill(0, 52)]
wkly_vaccinations_ceased = zeros(size(wkly_vaccinations))


chp_t2 = (Date(2022, 7, 23) - Date(2021, 12, 31)).value #Announcement of Public health emergency
inf_duration_red = 0.0

interventions_ensemble_novacs = [
    (
        trans_red2 = θ[9] * θ[11],
        vac_effectiveness = rand(Uniform(0.7, 0.85)),
        trans_red_other2 = θ[10] * θ[12],
        wkly_vaccinations = wkly_vaccinations_ceased,
        chp_t2,
        inf_duration_red,
    ) for θ in param_draws
]

mpx_sim_function_interventions = MonkeypoxUK.mpx_sim_function_interventions

preds_and_incidence_interventions_novac = map(
    (θ, intervention) ->
        mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention)[2:4],
    param_draws,
    interventions_ensemble_novacs,
)

preds_and_incidence_interventions_novac_4wkrev = map(
    (θ, intervention) ->
        mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention, 4)[2:4],
    param_draws,
    interventions_ensemble_novacs,
)

preds_and_incidence_interventions_novac_12wkrev = map(
    (θ, intervention) ->
        mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention, 12)[2:4],
    param_draws,
    interventions_ensemble_novacs,
)


##
d1, d2 = size(mpxv_wkly)

preds_novac = [x[1] for x in preds_and_incidence_interventions_novac]
cum_cases_forwards_novac =
    [cumsum(x[1][(d1+1):end, :], dims = 1) for x in preds_and_incidence_interventions_novac]
cred_int_novac = MonkeypoxUK.cred_intervals(preds_novac)

preds_novac_4wk = [x[1] for x in preds_and_incidence_interventions_novac_4wkrev]
cum_cases_forwards_novac = [
    cumsum(x[1][(d1+1):end, :], dims = 1) for
    x in preds_and_incidence_interventions_novac_4wkrev
]
cred_int_novac_4wk = MonkeypoxUK.cred_intervals(preds_novac_4wk)

preds_novac_12wk = [x[1] for x in preds_and_incidence_interventions_novac_12wkrev]
cum_cases_forwards_novac = [
    cumsum(x[1][(d1+1):end, :], dims = 1) for
    x in preds_and_incidence_interventions_novac_12wkrev
]
cred_int_novac_12wk = MonkeypoxUK.cred_intervals(preds_novac_12wk)
##

d_proj = 19

plt_msm_novac = plot(;
    ylabel = "Weekly cases",
    title = "UK Monkeypox Case Projections no vaccines (GBMSM)",# yscale=:log10,
    legend = :topright,
    # yticks=([1, 2, 11, 101, 1001], [0, 1, 10, 100, 1000]),
    ylims = (-5, 650),
    xticks = (
        [Date(2022, 5, 1) + Month(k) for k = 0:10],
        [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:10],
    ),
    left_margin = 5mm,
    size = (800, 600),
    dpi = 250,
    tickfont = 13,
    titlefont = 18,
    guidefont = 18,
    legendfont = 11,
)

plot!(
    plt_msm_novac,
    long_wks,
    cred_int_novac.mean_pred[:, 1],
    ribbon = (cred_int_novac.lb_pred_10[:, 1], cred_int_novac.ub_pred_10[:, 1]),
    lw = 3,
    color = 1,
    fillalpha = 0.2,
    lab = "No reversion",
)

plot!(
    plt_msm_novac,
    long_wks[d_proj:end],
    cred_int_novac_12wk.mean_pred[d_proj:end, 1],
    ribbon = (
        cred_int_novac_12wk.lb_pred_10[d_proj:end, 1],
        cred_int_novac_12wk.ub_pred_10[d_proj:end, 1],
    ),
    lw = 3,
    color = :black,
    fillalpha = 0.2,
    lab = "12 week reversion",
)

plot!(
    plt_msm_novac,
    long_wks[d_proj:end],
    cred_int_novac_4wk.mean_pred[d_proj:end, 1],
    ribbon = (
        cred_int_novac_4wk.lb_pred_10[d_proj:end, 1],
        cred_int_novac_4wk.ub_pred_10[d_proj:end, 1],
    ),
    lw = 3,
    color = 2,
    fillalpha = 0.2,
    lab = "4 week reversion",
)

scatter!(
    plt_msm_novac,
    wks[3:(end-1)],
    mpxv_wkly[3:(end-1), 1],
    lab = "Data",
    ms = 6,
    color = :black,
)
