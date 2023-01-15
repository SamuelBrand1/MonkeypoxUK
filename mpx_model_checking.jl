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

##Load posterior draws

param_draws = load("posteriors/posterior_param_draws_2022-10-17.jld2")["param_draws"]

## Public health emergency effect forecasts
n_lookaheadweeks = 26
long_wks = [wks; [wks[end] + Day(7 * k) for k = 1:n_lookaheadweeks]]
long_mpxv_wkly = [mpxv_wkly; zeros(n_lookaheadweeks, 2)]
wkly_vaccinations = [wkly_vaccinations; fill(0, 52)]
wkly_vaccinations_ceased = [copy(wkly_vaccinations)[1:length(wks)+1]; fill(0, 52)]

plt_vacs = plot(
    [wks[1] + Day(7 * (k - 1)) for k = 1:size(long_wks, 1)],
    cumsum(wkly_vaccinations)[1:size(long_mpxv_wkly, 1)],
    title = "Cumulative number of MPX vaccine doses",
    lab = "Projection",
    color = :black,
    lw = 3,# yticks=0:1000:8000,
    ylabel = "Cum. doses",
    xticks = (
        [Date(2022, 5, 1) + Month(k) for k = 0:12],
        [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:12],
    ),
    yticks = (0:10_000:50_000, string.(0:10_000:50_000)),
    size = (800, 600),
    left_margin = 5mm,
    guidefont = 16,
    tickfont = 13,
    titlefont = 18,
    legendfont = 16,
    legend = :right,
    right_margin = 5mm,
)


scatter!(
    plt_vacs,
    [Date(2022, 7, 1), Date(2022, 8, 30), Date(2022, 9, 22)],
    [0, 38_079, 40_426],
    lab = "UKHSA reported vaccines",
    ms = 8,
)
display(plt_vacs)
savefig(plt_vacs, "plots/vaccine_rollout.png")

vaccine_projections = DataFrame(
    "Date" => [wks[1] + Day(7 * (k - 1)) for k = 1:size(long_wks, 1)],
    "Projected cumulative vac doses" =>
        cumsum(wkly_vaccinations)[1:size(long_mpxv_wkly, 1)],
    "Cumulative vac doses (poor scenario)" =>
        cumsum(wkly_vaccinations_ceased)[1:size(long_wks, 1)],
)
CSV.write("projections/vaccine_rollout.csv", vaccine_projections)
##
chp_t2 = (Date(2022, 7, 23) - Date(2021, 12, 31)).value #Announcement of Public health emergency
inf_duration_red = 0.0

param_draws_no_behav = param_draws .|> θ -> [θ[1:(end-4)]; zeros(4)]

interventions_ensemble = [
    (
        trans_red2 = θ[9] * θ[11],
        vac_effectiveness = rand(Uniform(0.7, 0.85)),
        trans_red_other2 = θ[10] * θ[12],
        wkly_vaccinations,
        chp_t2,
        inf_duration_red,
    ) for θ in param_draws
]


no_vac_ensemble = [
    (
        trans_red2 = θ[9] * θ[11],#Based on posterior for first change point with extra dispersion
        vac_effectiveness = rand(Uniform(0.7, 0.85)),
        trans_red_other2 = θ[10] * θ[12],
        wkly_vaccinations = zeros(size(wkly_vaccinations_ceased)),
        chp_t2,
        inf_duration_red,
    ) for θ in param_draws
]

no_vac_and_no_red_ensemble = [
    (
        trans_red2 = 0,#Based on posterior for first change point with extra dispersion
        vac_effectiveness = rand(Uniform(0.7, 0.85)),
        trans_red_other2 = θ[10] * θ[12],
        wkly_vaccinations = zeros(size(wkly_vaccinations_ceased)),
        chp_t2,
        inf_duration_red,
    ) for θ in param_draws_no_behav
]

mpx_sim_function_interventions = MonkeypoxUK.mpx_sim_function_interventions

preds_and_incidence_interventions = map(
    (θ, intervention) ->
        mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention)[2:4],
    param_draws,
    interventions_ensemble,
)
preds_and_incidence_interventions_4wkrev = map(
    (θ, intervention) ->
        mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention, 4)[2:4],
    param_draws,
    interventions_ensemble,
)
preds_and_incidence_interventions_12wkrev = map(
    (θ, intervention) ->
        mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention, 12)[2:4],
    param_draws,
    interventions_ensemble,
)

preds_and_incidence_interventions_cvac = map(
    (θ, intervention) ->
        mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention)[2:4],
    param_draws,
    no_vac_ensemble,
)
preds_and_incidence_interventions_cvac_4wkrev = map(
    (θ, intervention) ->
        mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention, 4)[2:4],
    param_draws,
    no_vac_ensemble,
)
preds_and_incidence_interventions_cvac_12wkrev = map(
    (θ, intervention) ->
        mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention, 12)[2:4],
    param_draws,
    no_vac_ensemble,
)

preds_and_incidence_novac_no_chg = map(
    (θ, intervention) ->
        mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention)[2:4],
    param_draws_no_behav,
    no_vac_and_no_red_ensemble,
)

##Gather projections
d1, d2 = size(mpxv_wkly)

preds = [x[1] for x in preds_and_incidence_interventions]
preds_4wk = [x[1] for x in preds_and_incidence_interventions_4wkrev]
preds_12wk = [x[1] for x in preds_and_incidence_interventions_12wkrev]

preds_cvac = [x[1] for x in preds_and_incidence_interventions_cvac]
preds_cvac4wk = [x[1] for x in preds_and_incidence_interventions_cvac_4wkrev]
preds_cvac12wk = [x[1] for x in preds_and_incidence_interventions_cvac_12wkrev]

pred_unmitigated = [x[1] for x in preds_and_incidence_novac_no_chg]
cum_cases_unmitigated = [cumsum(x[1], dims = 1) for x in preds_and_incidence_novac_no_chg]

cum_cases_forwards =
    [cumsum(x[1][(d1+1):end, :], dims = 1) for x in preds_and_incidence_interventions]
cum_cases_forwards_4wk = [
    cumsum(x[1][(d1+1):end, :], dims = 1) for x in preds_and_incidence_interventions_4wkrev
]
cum_cases_forwards_12wk = [
    cumsum(x[1][(d1+1):end, :], dims = 1) for x in preds_and_incidence_interventions_12wkrev
]

cum_cases_forwards_cvac =
    [cumsum(x[1][(d1+1):end, :], dims = 1) for x in preds_and_incidence_interventions_cvac]
cum_cases_forwards_cvac4wk = [
    cumsum(x[1][(d1+1):end, :], dims = 1) for
    x in preds_and_incidence_interventions_cvac_4wkrev
]
cum_cases_forwards_cvac12wk = [
    cumsum(x[1][(d1+1):end, :], dims = 1) for
    x in preds_and_incidence_interventions_cvac_12wkrev
]


##Simulation projections

cred_int = MonkeypoxUK.cred_intervals(preds)
cred_int_4wk = MonkeypoxUK.cred_intervals(preds_4wk)
cred_int_12wk = MonkeypoxUK.cred_intervals(preds_12wk)

cred_int_cvac = MonkeypoxUK.cred_intervals(preds_cvac)
cred_int_cvac4wk = MonkeypoxUK.cred_intervals(preds_cvac4wk)
cred_int_cvac12wk = MonkeypoxUK.cred_intervals(preds_cvac12wk)

cred_int_unmitigated = MonkeypoxUK.cred_intervals(pred_unmitigated)
cred_int_cum_cases_unmitigated = MonkeypoxUK.cred_intervals(cum_cases_unmitigated)

## MSM projections
d_proj = 19
gbmsm_case_projections = DataFrame()
gbmsm_case_projections[:, "Date"] = long_wks
reported_gbmsm_cases = convert(
    Vector{Union{Float64,String}},
    fill("NA", length(cred_int_cvac4wk.mean_pred[:, 1])),
)
reported_gbmsm_cases[3:size(mpxv_wkly, 1)] .= mpxv_wkly[3:end, 1]
gbmsm_case_projections[:, "Inferred GBMSM cases"] = reported_gbmsm_cases

gbmsm_case_projections[:, "Projected GBMSM cases (post. mean; no reversion)"] =
    cred_int.mean_pred[:, 1]
gbmsm_case_projections[:, "Projected GBMSM cases (post. 10%; no reversion)"] =
    cred_int.mean_pred[:, 1] .- cred_int.lb_pred_10[:, 1]
gbmsm_case_projections[:, "Projected GBMSM cases (post. 90%; no reversion)"] =
    cred_int.mean_pred[:, 1] .+ cred_int.ub_pred_10[:, 1]

plt_msm = plot(;
    ylabel = "Weekly cases",
    title = "UK Monkeypox Case Projections (GBMSM)",# yscale=:log10,
    legend = :topright,
    # yticks=([1, 2, 11, 101, 1001], [0, 1, 10, 100, 1000]),
    ylims = (-5, 700),
    xticks = (
        [Date(2022, 5, 1) + Month(k) for k = 0:11],
        [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:11],
    ),
    left_margin = 5mm,
    size = (800, 600),
    dpi = 250,
    tickfont = 13,
    titlefont = 22,
    guidefont = 18,
    legendfont = 11,
)

plot!(
    plt_msm,
    long_wks,
    cred_int_12wk.mean_pred[:, 1],
    ribbon = (cred_int_12wk.lb_pred_10[:, 1], cred_int_12wk.ub_pred_10[:, 1]),
    lw = 3,
    color = :black,
    fillalpha = 0.2,
    lab = "12 week reversion",
)

gbmsm_case_projections[:, "Date"] = long_wks
gbmsm_case_projections[:, "Projected GBMSM cases (post. mean; 12 wk reversion)"] =
    cred_int_12wk.mean_pred[:, 1]
gbmsm_case_projections[:, "Projected GBMSM cases (post. 10%; 12 wk reversion)"] =
    cred_int_12wk.mean_pred[:, 1] .- cred_int_12wk.lb_pred_10[:, 1]
gbmsm_case_projections[:, "Projected GBMSM cases (post. 90%; 12 wk reversion)"] =
    cred_int_12wk.mean_pred[:, 1] .+ cred_int_12wk.ub_pred_10[:, 1]


plot!(
    plt_msm,
    long_wks[1:end],
    cred_int_cvac12wk.mean_pred[1:end, 1],
    # ribbon=(cred_int_cvac.lb_pred_10[d_proj:end, 1], cred_int_cvac.ub_pred_10[d_proj:end, 1]),
    lw = 3,
    color = :black,
    ls = :dash,
    fillalpha = 0.2,
    lab = "12 week reversion (no vaccines)",
)

gbmsm_case_projections[:, "Projected GBMSM cases (post. mean; 12 wk reversion + no vacs)"] =
    cred_int_cvac12wk.mean_pred[:, 1]
gbmsm_case_projections[:, "Projected GBMSM cases (post. 10%; 12 wk reversion + no vacs)"] =
    cred_int_cvac12wk.mean_pred[:, 1] .- cred_int_cvac12wk.lb_pred_10[:, 1]
gbmsm_case_projections[:, "Projected GBMSM cases (post. 90%; 12 wk reversion + no vacs)"] =
    cred_int_cvac12wk.mean_pred[:, 1] .+ cred_int_cvac12wk.ub_pred_10[:, 1]


plot!(
    plt_msm,
    long_wks[d_proj:end],
    cred_int_4wk.mean_pred[d_proj:end, 1],
    ribbon = (
        cred_int_4wk.lb_pred_10[d_proj:end, 1],
        cred_int_4wk.ub_pred_10[d_proj:end, 1],
    ),
    lw = 3,
    color = 2,
    fillalpha = 0.2,
    lab = "4 week reversion",
)

gbmsm_case_projections[:, "Projected GBMSM cases (post. mean; 4 wk reversion)"] =
    cred_int_4wk.mean_pred[:, 1]
gbmsm_case_projections[:, "Projected GBMSM cases (post. 10%; 4 wk reversion)"] =
    cred_int_4wk.mean_pred[:, 1] .- cred_int_4wk.lb_pred_10[:, 1]
gbmsm_case_projections[:, "Projected GBMSM cases (post. 90%; 4 wk reversion)"] =
    cred_int_4wk.mean_pred[:, 1] .+ cred_int_4wk.ub_pred_10[:, 1]



plot!(
    plt_msm,
    long_wks[19:end],
    cred_int_cvac4wk.mean_pred[19:end, 1],
    # ribbon=(cred_int_4wk.lb_pred_10[19:end, 1], cred_int_4wk.ub_pred_10[19:end, 1]),
    lw = 3,
    ls = :dash,
    color = 2,
    fillalpha = 0.2,
    lab = "4 week reversion (no vaccines)",
)

gbmsm_case_projections[:, "Projected GBMSM cases (post. mean; 4 wk reversion + no vacs)"] =
    cred_int_cvac4wk.mean_pred[:, 1]
gbmsm_case_projections[:, "Projected GBMSM cases (post. 10%; 4 wk reversion + no vacs)"] =
    cred_int_cvac4wk.mean_pred[:, 1] .- cred_int_cvac4wk.lb_pred_10[:, 1]
gbmsm_case_projections[:, "Projected GBMSM cases (post. 90%; 4 wk reversion + no vacs)"] =
    cred_int_cvac4wk.mean_pred[:, 1] .+ cred_int_cvac4wk.ub_pred_10[:, 1]



scatter!(
    plt_msm,
    wks[(end):end],
    mpxv_wkly[(end):end, 1],
    lab = "",
    ms = 6,
    color = :black,
    shape = :square,
)

scatter!(
    plt_msm,
    wks[3:(end-1)],
    mpxv_wkly[3:(end-1), 1],
    yerrors = (
        mpxv_wkly[3:(end-1), 1] .- lwr_mpxv_wkly[3:(end-1), 1],
        upr_mpxv_wkly[3:(end-1), 1] .- mpxv_wkly[3:(end-1), 1],
    ),
    lab = "Data",
    ms = 6,
    color = :black,
)


CSV.write(
    "projections/gbmsm_case_projections" * string(wks[end]) * ".csv",
    gbmsm_case_projections,
)
display(plt_msm)

##
nongbmsm_case_projections = DataFrame()
nongbmsm_case_projections[:, "Date"] = long_wks

reported_nongbmsm_cases = convert(
    Vector{Union{Float64,String}},
    fill("NA", length(cred_int_cvac4wk.mean_pred[:, 1])),
)
reported_nongbmsm_cases[3:size(mpxv_wkly, 1)] .= mpxv_wkly[3:end, 2]
nongbmsm_case_projections[:, "Inferred non-GBMSM cases"] = reported_nongbmsm_cases

nongbmsm_case_projections[:, "Projected non-GBMSM cases (post. mean; no reversion)"] =
    cred_int.mean_pred[:, 2]
nongbmsm_case_projections[:, "Projected non-GBMSM cases (post. 10%; no reversion)"] =
    cred_int.mean_pred[:, 2] .- cred_int.lb_pred_10[:, 2]
nongbmsm_case_projections[:, "Projected non-GBMSM cases (post. 90%; no reversion)"] =
    cred_int.mean_pred[:, 2] .+ cred_int.ub_pred_10[:, 2]



plt_nmsm = plot(;
    ylabel = "Weekly cases",
    title = "UK Monkeypox Case Projections (non-GBMSM)",# yscale=:log10,
    legend = :topright,
    # yticks=([1, 2, 11, 101, 1001], [0, 1, 10, 100, 1000]),
    ylims = (-1, 100),
    xticks = (
        [Date(2022, 5, 1) + Month(k) for k = 0:11],
        [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:11],
    ),
    left_margin = 5mm,
    size = (800, 600),
    dpi = 250,
    tickfont = 13,
    titlefont = 20,
    guidefont = 18,
    legendfont = 11,
)

plot!(
    plt_nmsm,
    long_wks,
    cred_int_12wk.mean_pred[:, 2],
    ribbon = (cred_int_12wk.lb_pred_10[:, 2], cred_int_12wk.ub_pred_10[:, 2]),
    lw = 3,
    color = :black,
    fillalpha = 0.2,
    lab = "12 week reversion",
)

nongbmsm_case_projections[:, "Projected non-GBMSM cases (post. mean; 12wk reversion)"] =
    cred_int_12wk.mean_pred[:, 2]
nongbmsm_case_projections[:, "Projected non-GBMSM cases (post. 10%; 12wk reversion)"] =
    cred_int_12wk.mean_pred[:, 2] .- cred_int_12wk.lb_pred_10[:, 2]
nongbmsm_case_projections[:, "Projected non-GBMSM cases (post. 90%; 12wk reversion)"] =
    cred_int_12wk.mean_pred[:, 2] .+ cred_int_12wk.ub_pred_10[:, 2]



plot!(
    plt_nmsm,
    long_wks[d_proj:end],
    cred_int_cvac12wk.mean_pred[d_proj:end, 2],
    # ribbon=(cred_int_cvac.lb_pred_10[d_proj:end, 1], cred_int_cvac.ub_pred_10[d_proj:end, 1]),
    lw = 3,
    color = :black,
    ls = :dash,
    fillalpha = 0.2,
    lab = "12 week reversion (no vaccines)",
)


nongbmsm_case_projections[
    :,
    "Projected non-GBMSM cases (post. mean; 12wk reversion + no vacs)",
] = cred_int_cvac12wk.mean_pred[:, 2]
nongbmsm_case_projections[
    :,
    "Projected non-GBMSM cases (post. 10%; 12wk reversion + no vacs)",
] = cred_int_cvac12wk.mean_pred[:, 2] .- cred_int_cvac12wk.lb_pred_10[:, 2]
nongbmsm_case_projections[
    :,
    "Projected non-GBMSM cases (post. 90%; 12wk reversion + no vacs)",
] = cred_int_cvac12wk.mean_pred[:, 2] .+ cred_int_cvac12wk.ub_pred_10[:, 2]


plot!(
    plt_nmsm,
    long_wks[19:end],
    cred_int_4wk.mean_pred[19:end, 2],
    ribbon = (cred_int_4wk.lb_pred_10[19:end, 2], cred_int_4wk.ub_pred_10[19:end, 2]),
    lw = 3,
    color = 2,
    fillalpha = 0.2,
    lab = "4 week reversion",
)

nongbmsm_case_projections[:, "Projected non-GBMSM cases (post. mean; 4wk reversion)"] =
    cred_int_4wk.mean_pred[:, 2]
nongbmsm_case_projections[:, "Projected non-GBMSM cases (post. 10%; 4wk reversion)"] =
    cred_int_4wk.mean_pred[:, 2] .- cred_int_4wk.lb_pred_10[:, 2]
nongbmsm_case_projections[:, "Projected non-GBMSM cases (post. 90%; 4wk reversion)"] =
    cred_int_4wk.mean_pred[:, 2] .+ cred_int_4wk.ub_pred_10[:, 2]



plot!(
    plt_nmsm,
    long_wks[19:end],
    cred_int_cvac4wk.mean_pred[19:end, 2],
    # ribbon=(cred_int_4wk.lb_pred_10[19:end, 1], cred_int_4wk.ub_pred_10[19:end, 1]),
    lw = 3,
    ls = :dash,
    color = 2,
    fillalpha = 0.2,
    lab = "4 week reversion (no vaccines)",
)

nongbmsm_case_projections[
    :,
    "Projected non-GBMSM cases (post. mean; 4wk reversion + no vacs)",
] = cred_int_cvac4wk.mean_pred[:, 2]
nongbmsm_case_projections[
    :,
    "Projected non-GBMSM cases (post. 10%; 4wk reversion + no vacs)",
] = cred_int_cvac4wk.mean_pred[:, 2] .- cred_int_cvac4wk.lb_pred_10[:, 2]
nongbmsm_case_projections[
    :,
    "Projected non-GBMSM cases (post. 90%; 4wk reversion + no vacs)",
] = cred_int_cvac4wk.mean_pred[:, 2] .+ cred_int_cvac4wk.ub_pred_10[:, 2]


scatter!(
    plt_nmsm,
    wks[(end):end],
    mpxv_wkly[(end):end, 2],
    lab = "",
    ms = 6,
    color = :black,
    shape = :square,
)
scatter!(
    plt_nmsm,
    wks[3:(end-1)],
    mpxv_wkly[3:(end-1), 2],
    lab = "Data",
    ms = 6,
    color = :black,
    yerrors = (
        mpxv_wkly[3:(end-1), 2] .- lwr_mpxv_wkly[3:(end-1), 2],
        upr_mpxv_wkly[3:(end-1), 2] .- mpxv_wkly[3:(end-1), 2],
    ),
)
CSV.write(
    "projections/nongbmsm_case_projections" * string(wks[end]) * ".csv",
    nongbmsm_case_projections,
)
display(plt_nmsm)

##cumulative Incidence plots

cred_int_cum_incidence = MonkeypoxUK.cred_intervals(cum_cases_forwards)
cred_int_cum_incidence4wks = MonkeypoxUK.cred_intervals(cum_cases_forwards_4wk)
cred_int_cum_incidence12wks = MonkeypoxUK.cred_intervals(cum_cases_forwards_12wk)

cred_int_cum_incidence_cvac = MonkeypoxUK.cred_intervals(cum_cases_forwards_cvac)
cred_int_cum_incidence_cvac4wks = MonkeypoxUK.cred_intervals(cum_cases_forwards_cvac4wk)
cred_int_cum_incidence_cvac12wks = MonkeypoxUK.cred_intervals(cum_cases_forwards_cvac12wk)



total_cases = sum(mpxv_wkly, dims = 1)

gbmsm_cum_case_projections = DataFrame()
gbmsm_cum_case_projections[:, "Date"] = long_wks[((d1+1)):end]

gbmsm_cum_case_projections[:, "Projected Cum. GBMSM cases (post. mean; no reversion)"] =
    total_cases[:, 1] .+ cred_int_cum_incidence.mean_pred[:, 1]
gbmsm_cum_case_projections[:, "Projected GBMSM cases (post. 10%; no reversion)"] =
    total_cases[:, 1] .+ cred_int_cum_incidence.mean_pred[:, 1] .-
    cred_int_cum_incidence.lb_pred_10[:, 1]
gbmsm_cum_case_projections[:, "Projected GBMSM cases (post. 90%; no reversion)"] =
    total_cases[:, 1] .+ cred_int_cum_incidence.mean_pred[:, 1] .+
    cred_int_cum_incidence.ub_pred_10[:, 1]


plt_cm_msm = plot(;
    ylabel = "Cumulative cases",
    title = "UK Monkeypox cumulative case projections (GBMSM)",#yscale=:log10,
    legend = :topleft,
    # yticks=(0:2500:12500, 0:2500:100),
    xticks = (
        [Date(2022, 5, 1) + Month(k) for k = 0:11],
        [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:11],
    ),
    left_margin = 5mm,
    size = (800, 600),
    dpi = 250,
    tickfont = 15,
    titlefont = 16,
    guidefont = 24,
    legendfont = 11,
)

plot!(
    plt_cm_msm,
    long_wks[((d1+1)):end],
    total_cases[:, 1] .+ cred_int_cum_incidence12wks.mean_pred[:, 1],
    ribbon = (
        cred_int_cum_incidence12wks.lb_pred_10[:, 1],
        cred_int_cum_incidence12wks.ub_pred_10[:, 1],
    ),
    lw = 3,
    color = :black,
    fillalpha = 0.4,
    lab = "12 week reversion",
)

gbmsm_cum_case_projections[:, "Projected Cum. GBMSM cases (post. mean; 12wk reversion)"] =
    total_cases[:, 1] .+ cred_int_cum_incidence12wks.mean_pred[:, 1]
gbmsm_cum_case_projections[:, "Projected GBMSM cases (post. 10%; 12wk reversion)"] =
    total_cases[:, 1] .+ cred_int_cum_incidence12wks.mean_pred[:, 1] .-
    cred_int_cum_incidence12wks.lb_pred_10[:, 1]
gbmsm_cum_case_projections[:, "Projected GBMSM cases (post. 90%; 12wk reversion)"] =
    total_cases[:, 1] .+ cred_int_cum_incidence12wks.mean_pred[:, 1] .+
    cred_int_cum_incidence12wks.ub_pred_10[:, 1]


plot!(
    plt_cm_msm,
    long_wks[((d1+1)):end],
    total_cases[:, 1] .+ cred_int_cum_incidence_cvac12wks.mean_pred[:, 1],
    ribbon = (
        cred_int_cum_incidence_cvac12wks.lb_pred_10[:, 1],
        cred_int_cum_incidence_cvac12wks.ub_pred_10[:, 1],
    ),
    lw = 3,
    ls = :dash,
    color = :black,
    fillalpha = 0.2,
    fillstyle = :x,
    lab = "12 week reversion (no vaccines)",
)

gbmsm_cum_case_projections[
    :,
    "Projected Cum. GBMSM cases (post. mean; 12wk reversion + no vacs)",
] = total_cases[:, 1] .+ cred_int_cum_incidence_cvac12wks.mean_pred[:, 1]
gbmsm_cum_case_projections[
    :,
    "Projected GBMSM cases (post. 10%; 12wk reversion + no vacs)",
] =
    total_cases[:, 1] .+ cred_int_cum_incidence_cvac12wks.mean_pred[:, 1] .-
    cred_int_cum_incidence_cvac12wks.lb_pred_10[:, 1]
gbmsm_cum_case_projections[
    :,
    "Projected GBMSM cases (post. 90%; 12wk reversion + no vacs)",
] =
    total_cases[:, 1] .+ cred_int_cum_incidence_cvac12wks.mean_pred[:, 1] .+
    cred_int_cum_incidence_cvac12wks.ub_pred_10[:, 1]



plot!(
    plt_cm_msm,
    long_wks[((d1+1)):end],
    total_cases[:, 1] .+ cred_int_cum_incidence4wks.mean_pred[:, 1],
    ribbon = (
        cred_int_cum_incidence4wks.lb_pred_10[:, 1],
        cred_int_cum_incidence4wks.ub_pred_10[:, 1],
    ),
    lw = 3,
    color = 2,
    fillalpha = 0.4,
    lab = "4 week reversion",
)

gbmsm_cum_case_projections[:, "Projected Cum. GBMSM cases (post. mean; 4wk reversion)"] =
    total_cases[:, 1] .+ cred_int_cum_incidence4wks.mean_pred[:, 1]
gbmsm_cum_case_projections[:, "Projected GBMSM cases (post. 10%; 4wk reversion)"] =
    total_cases[:, 1] .+ cred_int_cum_incidence4wks.mean_pred[:, 1] .-
    cred_int_cum_incidence4wks.lb_pred_10[:, 1]
gbmsm_cum_case_projections[:, "Projected GBMSM cases (post. 90%; 4wk reversion)"] =
    total_cases[:, 1] .+ cred_int_cum_incidence4wks.mean_pred[:, 1] .+
    cred_int_cum_incidence4wks.ub_pred_10[:, 1]


plot!(
    plt_cm_msm,
    long_wks[((d1+1)):end],
    total_cases[:, 1] .+ cred_int_cum_incidence_cvac4wks.mean_pred[:, 1],
    ribbon = (
        cred_int_cum_incidence_cvac4wks.lb_pred_10[:, 1],
        cred_int_cum_incidence_cvac4wks.ub_pred_10[:, 1],
    ),
    lw = 3,
    ls = :dash,
    fillstyle = :x,
    color = 2,
    fillalpha = 0.3,
    lab = "4 week reversion (no vaccines)",
)

gbmsm_cum_case_projections[
    :,
    "Projected Cum. GBMSM cases (post. mean; 4wk reversion + no vacs)",
] = total_cases[:, 1] .+ cred_int_cum_incidence_cvac4wks.mean_pred[:, 1]
gbmsm_cum_case_projections[
    :,
    "Projected GBMSM cases (post. 10%; 4wk reversion + no vacs)",
] =
    total_cases[:, 1] .+ cred_int_cum_incidence_cvac4wks.mean_pred[:, 1] .-
    cred_int_cum_incidence_cvac4wks.lb_pred_10[:, 1]
gbmsm_cum_case_projections[
    :,
    "Projected GBMSM cases (post. 90%; 4wk reversion + no vacs)",
] =
    total_cases[:, 1] .+ cred_int_cum_incidence_cvac4wks.mean_pred[:, 1] .+
    cred_int_cum_incidence_cvac4wks.ub_pred_10[:, 1]

plot!(
    plt_cm_msm,
    long_wks,
    cred_int_cum_cases_unmitigated.mean_pred[:, 1],
    ribbon = (
        cred_int_cum_cases_unmitigated.lb_pred_10[:, 1],
        cred_int_cum_cases_unmitigated.ub_pred_10[:, 1],
    ),
    lw = 3,
    ls = :dash,
    color = :green,
    fillalpha = 0.2,
    fillstyle = :x,
    lab = "Unmitigated",
)


scatter!(
    plt_cm_msm,
    wks,
    cumsum(mpxv_wkly[:, 1], dims = 1),
    lab = "Data",
    ms = 6,
    color = :black,
)

CSV.write(
    "projections/gbmsm_cumulative_case_projections" * string(wks[end]) * ".csv",
    gbmsm_cum_case_projections,
)
display(plt_cm_msm)
savefig(plt_cm_msm, "plt_cm_msm.svg")

##
total_cases = sum(mpxv_wkly, dims = 1)

nongbmsm_cum_case_projections = DataFrame()
nongbmsm_cum_case_projections[:, "Date"] = long_wks[((d1+1)):end]

nongbmsm_cum_case_projections[
    :,
    "Projected Cum. non-GBMSM cases (post. mean; no reversion)",
] = total_cases[:, 2] .+ cred_int_cum_incidence.mean_pred[:, 2]
nongbmsm_cum_case_projections[:, "Projected non-GBMSM cases (post. 10%; no reversion)"] =
    total_cases[:, 2] .+ cred_int_cum_incidence.mean_pred[:, 2] .-
    cred_int_cum_incidence.lb_pred_10[:, 2]
nongbmsm_cum_case_projections[:, "Projected non-GBMSM cases (post. 90%; no reversion)"] =
    total_cases[:, 2] .+ cred_int_cum_incidence.mean_pred[:, 2] .+
    cred_int_cum_incidence.ub_pred_10[:, 2]



plt_cm_nmsm = plot(;
    ylabel = "Cumulative cases",
    title = "UK Monkeypox cumulative case projections (non-GBMSM)",#yscale=:log10,
    legend = :topleft,
    # yticks=(0:2500:12500, 0:2500:12500),
    xticks = (
        [Date(2022, 5, 1) + Month(k) for k = 0:11],
        [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:11],
    ),
    left_margin = 5mm,
    size = (800, 600),
    dpi = 250,
    tickfont = 15,
    titlefont = 16,
    guidefont = 24,
    legendfont = 11,
)

plot!(
    plt_cm_nmsm,
    long_wks[((d1+1)):end],
    total_cases[:, 2] .+ cred_int_cum_incidence12wks.mean_pred[:, 2],
    ribbon = (
        cred_int_cum_incidence12wks.lb_pred_10[:, 2],
        cred_int_cum_incidence12wks.ub_pred_10[:, 2],
    ),
    lw = 3,
    color = :black,
    fillalpha = 0.4,
    lab = "12 week reversion",
)

nongbmsm_cum_case_projections[
    :,
    "Projected Cum. non-GBMSM cases (post. mean; 12wk reversion)",
] = total_cases[:, 2] .+ cred_int_cum_incidence12wks.mean_pred[:, 2]
nongbmsm_cum_case_projections[:, "Projected non-GBMSM cases (post. 10%; 12wk reversion)"] =
    total_cases[:, 2] .+ cred_int_cum_incidence12wks.mean_pred[:, 2] .-
    cred_int_cum_incidence12wks.lb_pred_10[:, 2]
nongbmsm_cum_case_projections[:, "Projected non-GBMSM cases (post. 90%; 12wk reversion)"] =
    total_cases[:, 2] .+ cred_int_cum_incidence12wks.mean_pred[:, 2] .+
    cred_int_cum_incidence12wks.ub_pred_10[:, 2]


plot!(
    plt_cm_nmsm,
    long_wks[((d1+1)):end],
    total_cases[:, 2] .+ cred_int_cum_incidence_cvac12wks.mean_pred[:, 2],
    ribbon = (
        cred_int_cum_incidence_cvac12wks.lb_pred_10[:, 2],
        cred_int_cum_incidence_cvac12wks.ub_pred_10[:, 2],
    ),
    lw = 3,
    ls = :dash,
    color = :black,
    fillalpha = 0.2,
    fillstyle = :x,
    lab = "12 week reversion (no vaccines)",
)

nongbmsm_cum_case_projections[
    :,
    "Projected Cum. non-GBMSM cases (post. mean; 12wk reversion + no vacs)",
] = total_cases[:, 2] .+ cred_int_cum_incidence_cvac12wks.mean_pred[:, 2]
nongbmsm_cum_case_projections[
    :,
    "Projected non-GBMSM cases (post. 10%; 12wk reversion + no vacs)",
] =
    total_cases[:, 2] .+ cred_int_cum_incidence_cvac12wks.mean_pred[:, 2] .-
    cred_int_cum_incidence_cvac12wks.lb_pred_10[:, 2]
nongbmsm_cum_case_projections[
    :,
    "Projected non-GBMSM cases (post. 90%; 12wk reversion + no vacs)",
] =
    total_cases[:, 2] .+ cred_int_cum_incidence_cvac12wks.mean_pred[:, 2] .+
    cred_int_cum_incidence_cvac12wks.ub_pred_10[:, 2]



plot!(
    plt_cm_nmsm,
    long_wks[((d1+1)):end],
    total_cases[:, 2] .+ cred_int_cum_incidence4wks.mean_pred[:, 2],
    ribbon = (
        cred_int_cum_incidence4wks.lb_pred_10[:, 2],
        cred_int_cum_incidence4wks.ub_pred_10[:, 2],
    ),
    lw = 3,
    color = 2,
    fillalpha = 0.4,
    lab = "4 week reversion",
)

nongbmsm_cum_case_projections[
    :,
    "Projected Cum. non-GBMSM cases (post. mean; 4wk reversion)",
] = total_cases[:, 2] .+ cred_int_cum_incidence4wks.mean_pred[:, 2]
nongbmsm_cum_case_projections[:, "Projected non-GBMSM cases (post. 10%; 4wk reversion)"] =
    total_cases[:, 2] .+ cred_int_cum_incidence4wks.mean_pred[:, 2] .-
    cred_int_cum_incidence4wks.lb_pred_10[:, 2]
nongbmsm_cum_case_projections[:, "Projected non-GBMSM cases (post. 90%; 4wk reversion)"] =
    total_cases[:, 2] .+ cred_int_cum_incidence4wks.mean_pred[:, 2] .+
    cred_int_cum_incidence4wks.ub_pred_10[:, 2]


plot!(
    plt_cm_nmsm,
    long_wks[((d1+1)):end],
    total_cases[:, 2] .+ cred_int_cum_incidence_cvac4wks.mean_pred[:, 2],
    ribbon = (
        cred_int_cum_incidence_cvac4wks.lb_pred_10[:, 2],
        cred_int_cum_incidence_cvac4wks.ub_pred_10[:, 2],
    ),
    lw = 3,
    ls = :dash,
    color = 2,
    fillalpha = 0.2,
    fillstyle = :x,
    lab = "4 week reversion (no vaccines)",
)

nongbmsm_cum_case_projections[
    :,
    "Projected Cum. non-GBMSM cases (post. mean; 4wk reversion + no vacs)",
] = total_cases[:, 2] .+ cred_int_cum_incidence_cvac4wks.mean_pred[:, 2]
nongbmsm_cum_case_projections[
    :,
    "Projected non-GBMSM cases (post. 10%; 4wk reversion + no vacs)",
] =
    total_cases[:, 2] .+ cred_int_cum_incidence_cvac4wks.mean_pred[:, 2] .-
    cred_int_cum_incidence_cvac4wks.lb_pred_10[:, 2]
nongbmsm_cum_case_projections[
    :,
    "Projected non-GBMSM cases (post. 90%; 4wk reversion + no vacs)",
] =
    total_cases[:, 2] .+ cred_int_cum_incidence_cvac4wks.mean_pred[:, 2] .+
    cred_int_cum_incidence_cvac4wks.ub_pred_10[:, 2]


plot!(
    plt_cm_nmsm,
    long_wks,
    cred_int_cum_cases_unmitigated.mean_pred[:, 2],
    ribbon = (
        cred_int_cum_cases_unmitigated.lb_pred_10[:, 2],
        cred_int_cum_cases_unmitigated.ub_pred_10[:, 2],
    ),
    lw = 3,
    ls = :dash,
    color = :green,
    fillalpha = 0.2,
    fillstyle = :x,
    lab = "Unmitigated",
)


scatter!(
    plt_cm_nmsm,
    wks,
    cumsum(mpxv_wkly[:, 2], dims = 1),
    lab = "Data",
    ms = 6,
    color = :black,
)

CSV.write(
    "projections/nongbmsm_cumulative_case_projections" * string(wks[end]) * ".csv",
    nongbmsm_cum_case_projections,
)

display(plt_cm_nmsm)
savefig(plt_cm_nmsm, "plt_cm_nmsm.svg")
##Combined plot for cases
lo = @layout [a b; c d]
plt = plot(
    plt_msm,
    plt_nmsm,
    plt_cm_msm,
    plt_cm_nmsm,
    size = (1600, 1200),
    dpi = 250,
    left_margin = 10mm,
    bottom_margin = 10mm,
    right_margin = 10mm,
    layout = lo,
)
display(plt)


savefig(plt, "plots/case_projections_" * string(wks[end]) * ".png")
savefig(plt, "plots/case_projections_" * string(wks[end]) * ".svg")


## Change in transmission over time

prob_trans = [θ[4] for θ in param_draws]
mean_inf_periods = [θ[3] for θ in param_draws]
red_sx_trans = [θ[9] for θ in param_draws]
chp1 = [θ[8] for θ in param_draws]
red_sx_trans2 = [int.trans_red2 for int in interventions_ensemble]

"""
    function generate_trans_risk_over_time(p_trans, trans_red, trans_red2, chp, chp2, ts)

Calculate the transmission risk daily.
"""
function generate_trans_risk_over_time(p_trans, trans_red, trans_red2, chp, chp2, ts)
    log_p = [
        log(p_trans) + log(1 - trans_red) * (t >= chp) + log(1 - trans_red2) * (t >= chp2) for t in ts
    ]
    return exp.(log_p)
end

"""
function generate_trans_risk_over_time(
    p_trans,
    trans_red,
    trans_red2,
    chp,
    chp2,
    ts,
    days_reversion;
    reversion_time=(Date(2022, 9, 1) - Date(2021, 12, 31)).value
)

Calculate the transmission risk daily, with a 1% -> 99% reversion to normal occuring over `days_reversion`.
"""
function generate_trans_risk_over_time(
    p_trans,
    trans_red,
    trans_red2,
    chp,
    chp2,
    ts,
    days_reversion;
    reversion_time = (Date(2022, 9, 1) - Date(2021, 12, 31)).value,
)
    # log_r = -(log(1 - trans_red) + log(1 - trans_red2)) / days_reversion
    T₅₀ = reversion_time + (days_reversion / 2) # 50% return to normal point
    κ = (days_reversion / 2) / 4.6 # logistic scale for return to normal: 4.6 is κ = 1 time to go from 0.01 to 0.5 and 0.5 to 0.99
    p_min = p_trans * (1 - trans_red) * (1 - trans_red2)

    log_p = [
        log(p_trans) + log(1 - trans_red) * (t >= chp) + log(1 - trans_red2) * (t >= chp2) for t in ts
    ]

    return exp.(log_p) .+ [(p_trans - p_min) * sigmoid((t - T₅₀) / κ) for t in ts]
end

"""
function generate_R_gbmsm_over_time(
    p_trans,
    mean_inf_period,
    trans_red,
    trans_red2,
    chp,
    chp2,
    ts,
    days_reversion;
    av_cnt_rate = mean(mean_daily_cnts),
    reversion_time=(Date(2022, 9, 1) - Date(2021, 12, 31)).value
)

Calculate the R₀ for GBMSM only, with a 1% -> 99% reversion to normal occuring over `days_reversion`.
"""
function generate_R_gbmsm_over_time(
    p_trans,
    mean_inf_period,
    trans_red,
    trans_red2,
    chp,
    chp2,
    ts,
    days_reversion;
    av_cnt_rate = mean(mean_daily_cnts),
    reversion_time = (Date(2022, 9, 1) - Date(2021, 12, 31)).value,
)
    # log_r = -(log(1 - trans_red) + log(1 - trans_red2)) / days_reversion
    T₅₀ = reversion_time + (days_reversion / 2) # 50% return to normal point
    κ = (days_reversion / 2) / 4.6 # logistic scale for return to normal: 4.6 is κ = 1 time to go from 0.01 to 0.5 and 0.5 to 0.99
    p_min = p_trans * (1 - trans_red) * (1 - trans_red2)

    log_p = [
        log(p_trans) + log(1 - trans_red) * (t >= chp) + log(1 - trans_red2) * (t >= chp2) for t in ts
    ]
    pₜ = exp.(log_p) .+ [(p_trans - p_min) * sigmoid((t - T₅₀) / κ) for t in ts]
    return mean_inf_period .* av_cnt_rate .* pₜ
end

##


ts_risk = 1:2*365
p_sx_trans_risks =
    map(
        (p_tr, red_sx_tr, red_sx_tr2, ch1) -> generate_trans_risk_over_time(
            p_tr,
            red_sx_tr,
            red_sx_tr2,
            ch1,
            chp_t2,
            ts_risk,
        ),
        prob_trans,
        red_sx_trans,
        red_sx_trans2,
        chp1,
    ) .|> x -> reshape(x, length(x), 1)


p_sx_trans_risks_4wk =
    map(
        (p_tr, red_sx_tr, red_sx_tr2, ch1) -> generate_trans_risk_over_time(
            p_tr,
            red_sx_tr,
            red_sx_tr2,
            ch1,
            chp_t2,
            ts_risk,
            4 * 7,
        ),
        prob_trans,
        red_sx_trans,
        red_sx_trans2,
        chp1,
    ) .|> x -> reshape(x, length(x), 1)

R_gbmsm_4wk =
    map(
        (p_tr, mean_inf_period, red_sx_tr, red_sx_tr2, ch1) -> generate_R_gbmsm_over_time(
            p_tr,
            mean_inf_period,
            red_sx_tr,
            red_sx_tr2,
            ch1,
            chp_t2,
            ts_risk,
            4 * 7,
        ),
        prob_trans,
        mean_inf_periods,
        red_sx_trans,
        red_sx_trans2,
        chp1,
    ) .|> x -> reshape(x, length(x), 1)

p_sx_trans_risks_12wk =
    map(
        (p_tr, red_sx_tr, red_sx_tr2, ch1) -> generate_trans_risk_over_time(
            p_tr,
            red_sx_tr,
            red_sx_tr2,
            ch1,
            chp_t2,
            ts_risk,
            12 * 7,
        ),
        prob_trans,
        red_sx_trans,
        red_sx_trans2,
        chp1,
    ) .|> x -> reshape(x, length(x), 1)

R_gbmsm_12wk =
    map(
        (p_tr, mean_inf_period, red_sx_tr, red_sx_tr2, ch1) -> generate_R_gbmsm_over_time(
            p_tr,
            mean_inf_period,
            red_sx_tr,
            red_sx_tr2,
            ch1,
            chp_t2,
            ts_risk,
            12 * 7,
        ),
        prob_trans,
        mean_inf_periods,
        red_sx_trans,
        red_sx_trans2,
        chp1,
    ) .|> x -> reshape(x, length(x), 1)

sx_trans_risk_cred_int = prev_cred_intervals(p_sx_trans_risks)
sx_trans_risk_cred_int_4wk = prev_cred_intervals(p_sx_trans_risks_4wk)
sx_trans_risk_cred_int_12wk = prev_cred_intervals(p_sx_trans_risks_12wk)
R_gbmsms_cred_int_4wk = prev_cred_intervals(R_gbmsm_4wk)
R_gbmsms_cred_int_12wk = prev_cred_intervals(R_gbmsm_12wk)


dates = [Date(2021, 12, 31) + Day(t) for t in ts_risk]
f = findfirst(dates .== Date(2022, 7, 23))

#Posterior probability of >10% decrease in risk
p_sx_risk_pheic = mean([p_trans[f] < p_trans[1] * 0.9 for p_trans in p_sx_trans_risks])

sx_cnt_trans_risk = DataFrame()
sx_cnt_trans_risk[:, "Date"] = dates[dates.>=Date(2022, 5, 1)]

R_gbmsm_df = DataFrame()
R_gbmsm_df[:, "Date"] = dates[dates.>=Date(2022, 5, 1)]



plt_chng = plot(
    dates,
    sx_trans_risk_cred_int_12wk.mean_pred,
    ribbon = (
        sx_trans_risk_cred_int_12wk.lb_pred_10,
        sx_trans_risk_cred_int_12wk.ub_pred_10,
    ),
    lw = 3,
    fillalpha = 0.2,
    lab = "Transmission probability (12 week reversion)",
    title = "Transmission probability (sexual contacts)",
    ylabel = "Prob. per sexual contact",
    xlims = (long_wks[1] - Day(7), long_wks[end] + Day(7)),
    ylims = (0, 0.7),
    xticks = (
        [Date(2022, 5, 1) + Month(k) for k = 0:11],
        [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:11],
    ),
    color = :black,
    left_margin = 5mm,
    size = (800, 600),
    dpi = 250,
    tickfont = 13,
    titlefont = 22,
    guidefont = 18,
    legendfont = 11,
)

plt_R_gbmsm = plot(
    dates,
    R_gbmsms_cred_int_12wk.mean_pred,
    ribbon = (R_gbmsms_cred_int_12wk.lb_pred_10, R_gbmsms_cred_int_12wk.ub_pred_10),
    lw = 3,
    fillalpha = 0.2,
    lab = "12 week reversion",
    title = "Reproductive number (GBMSM)",
    ylabel = "R₀(t) (GBMSM)",
    xlims = (long_wks[1] - Day(7), long_wks[end] + Day(7)),
    ylims = (0, 10.5),
    xticks = (
        [Date(2022, 5, 1) + Month(k) for k = 0:11],
        [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:11],
    ),
    yticks = 1:10,
    color = :black,
    left_margin = 5mm,
    size = (800, 600),
    dpi = 250,
    tickfont = 13,
    titlefont = 22,
    guidefont = 18,
    legendfont = 11,
)


sx_cnt_trans_risk[:, "Risk trans. per sx cnt (post. mean, no reversion)"] =
    sx_trans_risk_cred_int.mean_pred[dates.>=Date(2022, 5, 1)]
sx_cnt_trans_risk[:, "Risk trans. per sx cnt (post. 10%, no reversion)"] =
    sx_trans_risk_cred_int.mean_pred[dates.>=Date(2022, 5, 1)] .-
    sx_trans_risk_cred_int.lb_pred_10[dates.>=Date(2022, 5, 1)]
sx_cnt_trans_risk[:, "Risk trans. per sx cnt (post. 90%, no reversion)"] =
    sx_trans_risk_cred_int.mean_pred[dates.>=Date(2022, 5, 1)] .+
    sx_trans_risk_cred_int.lb_pred_10[dates.>=Date(2022, 5, 1)]

plot!(
    plt_chng,
    dates[f:end],
    sx_trans_risk_cred_int_4wk.mean_pred[f:end],
    ribbon = (
        sx_trans_risk_cred_int_4wk.lb_pred_10[f:end],
        sx_trans_risk_cred_int_4wk.ub_pred_10[f:end],
    ),
    lw = 3,
    fillalpha = 0.3,
    lab = "Transmission probability (4 week reversion)",
)

plot!(
    plt_R_gbmsm,
    dates[f:end],
    R_gbmsms_cred_int_4wk.mean_pred[f:end],
    ribbon = (
        R_gbmsms_cred_int_4wk.lb_pred_10[f:end],
        R_gbmsms_cred_int_4wk.ub_pred_10[f:end],
    ),
    lw = 3,
    fillalpha = 0.3,
    lab = "4 week reversion",
)


sx_cnt_trans_risk[:, "Risk trans. per sx cnt (post. mean, 4wk reversion)"] =
    sx_trans_risk_cred_int_4wk.mean_pred[dates.>=Date(2022, 5, 1)]
sx_cnt_trans_risk[:, "Risk trans. per sx cnt (post. 10%, 4wk reversion)"] =
    sx_trans_risk_cred_int_4wk.mean_pred[dates.>=Date(2022, 5, 1)] .-
    sx_trans_risk_cred_int_4wk.lb_pred_10[dates.>=Date(2022, 5, 1)]
sx_cnt_trans_risk[:, "Risk trans. per sx cnt (post. 90%, 4wk reversion)"] =
    sx_trans_risk_cred_int_4wk.mean_pred[dates.>=Date(2022, 5, 1)] .+
    sx_trans_risk_cred_int_4wk.lb_pred_10[dates.>=Date(2022, 5, 1)]

R_gbmsm_df[:, "R0 GBMSM (post. mean, 4wk reversion)"] =
    R_gbmsms_cred_int_4wk.mean_pred[dates.>=Date(2022, 5, 1)]
R_gbmsm_df[:, "R0 GBMSM (post. 10%, 4wk reversion)"] =
    R_gbmsms_cred_int_4wk.mean_pred[dates.>=Date(2022, 5, 1)] .-
    R_gbmsms_cred_int_4wk.lb_pred_10[dates.>=Date(2022, 5, 1)]
R_gbmsm_df[:, "R0 GBMSM (post. 90%, 4wk reversion)"] =
    R_gbmsms_cred_int_4wk.mean_pred[dates.>=Date(2022, 5, 1)] .+
    R_gbmsms_cred_int_4wk.lb_pred_10[dates.>=Date(2022, 5, 1)]

sx_cnt_trans_risk[:, "Risk trans. per sx cnt (post. mean, 12wk reversion)"] =
    sx_trans_risk_cred_int_12wk.mean_pred[dates.>=Date(2022, 5, 1)]
sx_cnt_trans_risk[:, "Risk trans. per sx cnt (post. 10%, 12wk reversion)"] =
    sx_trans_risk_cred_int_12wk.mean_pred[dates.>=Date(2022, 5, 1)] .-
    sx_trans_risk_cred_int_12wk.lb_pred_10[dates.>=Date(2022, 5, 1)]
sx_cnt_trans_risk[:, "Risk trans. per sx cnt (post. 90%, 12wk reversion)"] =
    sx_trans_risk_cred_int_12wk.mean_pred[dates.>=Date(2022, 5, 1)] .+
    sx_trans_risk_cred_int_12wk.lb_pred_10[dates.>=Date(2022, 5, 1)]

R_gbmsm_df[:, "R0 GBMSM (post. mean, 12wk reversion)"] =
    R_gbmsms_cred_int_12wk.mean_pred[dates.>=Date(2022, 5, 1)]
R_gbmsm_df[:, "R0 GBMSM (post. 10%, 12wk reversion)"] =
    R_gbmsms_cred_int_12wk.mean_pred[dates.>=Date(2022, 5, 1)] .-
    R_gbmsms_cred_int_12wk.lb_pred_10[dates.>=Date(2022, 5, 1)]
R_gbmsm_df[:, "R0 GBMSM (post. 90%, 12wk reversion)"] =
    R_gbmsms_cred_int_12wk.mean_pred[dates.>=Date(2022, 5, 1)] .+
    R_gbmsms_cred_int_12wk.lb_pred_10[dates.>=Date(2022, 5, 1)]

vline!(
    plt_chng,
    [Date(2022, 7, 23)],
    lw = 3,
    color = :black,
    ls = :dot,
    lab = "",
    annotation = (Date(2022, 7, 23) + Day(7), 0.5, "WHO declaration"),
    annotationrotation = 270,
)

vline!(
    plt_R_gbmsm,
    [Date(2022, 7, 23)],
    lw = 3,
    color = :black,
    ls = :dot,
    lab = "",
    annotation = (Date(2022, 7, 23) + Day(7), 7.5, "WHO declaration"),
    annotationrotation = 270,
)
# annotate!(plt_chng,Date(2022, 7, 23),0.5, "WHO" )

display(plt_R_gbmsm)

CSV.write("projections/sx_cnt_risk" * string(wks[end]) * ".csv", sx_cnt_trans_risk)
CSV.write("projections/R_gbmsm" * string(wks[end]) * ".csv", R_gbmsm_df)

##
R0_other = [θ[5] for θ in param_draws]
red_oth_trans = [θ[10] for θ in param_draws]
chp1 = [θ[8] for θ in param_draws]
red_oth_trans2 = [int.trans_red_other2 for int in interventions_ensemble]




p_oth_trans_risks =
    map(
        (p_tr, red_sx_tr, red_sx_tr2, ch1) -> generate_trans_risk_over_time(
            p_tr,
            red_sx_tr,
            red_sx_tr2,
            ch1,
            chp_t2,
            ts_risk,
        ),
        R0_other,
        red_oth_trans,
        red_oth_trans2,
        chp1,
    ) .|> x -> reshape(x, length(x), 1)

p_oth_trans_risks_4wks =
    map(
        (p_tr, red_sx_tr, red_sx_tr2, ch1) -> generate_trans_risk_over_time(
            p_tr,
            red_sx_tr,
            red_sx_tr2,
            ch1,
            chp_t2,
            ts_risk,
            4 * 7,
        ),
        R0_other,
        red_oth_trans,
        red_oth_trans2,
        chp1,
    ) .|> x -> reshape(x, length(x), 1)

p_oth_trans_risks_12wks =
    map(
        (p_tr, red_sx_tr, red_sx_tr2, ch1) -> generate_trans_risk_over_time(
            p_tr,
            red_sx_tr,
            red_sx_tr2,
            ch1,
            chp_t2,
            ts_risk,
            12 * 7,
        ),
        R0_other,
        red_oth_trans,
        red_oth_trans2,
        chp1,
    ) .|> x -> reshape(x, length(x), 1)

oth_sx_trans_risk_cred_int = prev_cred_intervals(p_oth_trans_risks)
oth_sx_trans_risk_cred_int4wks = prev_cred_intervals(p_oth_trans_risks_4wks)
oth_sx_trans_risk_cred_int12wks = prev_cred_intervals(p_oth_trans_risks_12wks)

#Posterior probability of >10% decrease in risk
# p_oth_risk_dec = mean([p_trans[f] < p_trans[1]*0.9 for p_trans in p_oth_trans_risks ])

R_oth_risk = DataFrame()
R_oth_risk[:, "Date"] = dates[dates.>=Date(2022, 5, 1)]



plt_chng_oth = plot(
    dates,
    oth_sx_trans_risk_cred_int12wks.mean_pred,
    ribbon = (
        oth_sx_trans_risk_cred_int12wks.lb_pred_10,
        oth_sx_trans_risk_cred_int12wks.ub_pred_10,
    ),
    lw = 3,
    fillalpha = 0.2,
    lab = "R0, other contacts (12 week reversion)",
    title = "Reproductive number (other contacts)",
    ylabel = "R₀(t) (other contacts)",
    xlims = (long_wks[1] - Day(7), long_wks[end] + Day(7)),
    ylims = (0, 0.3),
    xticks = (
        [Date(2022, 5, 1) + Month(k) for k = 0:11],
        [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:11],
    ),
    color = :black,
    left_margin = 5mm,
    right_margin = 5mm,
    size = (800, 600),
    dpi = 250,
    tickfont = 13,
    titlefont = 24,
    guidefont = 18,
    legendfont = 11,
)

R_oth_risk[:, "R other (post. mean, no reversion)"] =
    oth_sx_trans_risk_cred_int.mean_pred[dates.>=Date(2022, 5, 1)]
R_oth_risk[:, "R other (post. 10%, no reversion)"] =
    oth_sx_trans_risk_cred_int.mean_pred[dates.>=Date(2022, 5, 1)] .-
    oth_sx_trans_risk_cred_int.lb_pred_10[dates.>=Date(2022, 5, 1)]
R_oth_risk[:, "R other (post. 90%, no reversion)"] =
    oth_sx_trans_risk_cred_int.mean_pred[dates.>=Date(2022, 5, 1)] .+
    oth_sx_trans_risk_cred_int.lb_pred_10[dates.>=Date(2022, 5, 1)]


plot!(
    plt_chng_oth,
    dates[f:end],
    oth_sx_trans_risk_cred_int4wks.mean_pred[f:end],
    ribbon = (
        oth_sx_trans_risk_cred_int4wks.lb_pred_10[f:end],
        oth_sx_trans_risk_cred_int4wks.ub_pred_10[f:end],
    ),
    lw = 3,
    fillalpha = 0.3,
    lab = "R0, other (4 week reversion)",
)

R_oth_risk[:, "R other (post. mean, 4wk reversion)"] =
    oth_sx_trans_risk_cred_int4wks.mean_pred[dates.>=Date(2022, 5, 1)]
R_oth_risk[:, "R other (post. 10%, 4wk reversion)"] =
    oth_sx_trans_risk_cred_int4wks.mean_pred[dates.>=Date(2022, 5, 1)] .-
    oth_sx_trans_risk_cred_int4wks.lb_pred_10[dates.>=Date(2022, 5, 1)]
R_oth_risk[:, "R other (post. 90%, 4wk reversion)"] =
    oth_sx_trans_risk_cred_int4wks.mean_pred[dates.>=Date(2022, 5, 1)] .+
    oth_sx_trans_risk_cred_int4wks.lb_pred_10[dates.>=Date(2022, 5, 1)]


# plot!(
#     plt_chng_oth,
#     dates[f:end],
#     oth_sx_trans_risk_cred_int12wks.mean_pred[f:end],
#     ribbon=(
#         oth_sx_trans_risk_cred_int12wks.lb_pred_10[f:end],
#         oth_sx_trans_risk_cred_int12wks.ub_pred_10[f:end],
#     ),
#     lw=3,
#     fillalpha=0.3,
#     color=:black,
#     lab="R0, other (12 week reversion)",
# )

R_oth_risk[:, "R other (post. mean, 12wk reversion)"] =
    oth_sx_trans_risk_cred_int12wks.mean_pred[dates.>=Date(2022, 5, 1)]
R_oth_risk[:, "R other (post. 10%, 12wk reversion)"] =
    oth_sx_trans_risk_cred_int12wks.mean_pred[dates.>=Date(2022, 5, 1)] .-
    oth_sx_trans_risk_cred_int12wks.lb_pred_10[dates.>=Date(2022, 5, 1)]
R_oth_risk[:, "R other (post. 90%, 12wk reversion)"] =
    oth_sx_trans_risk_cred_int12wks.mean_pred[dates.>=Date(2022, 5, 1)] .+
    oth_sx_trans_risk_cred_int12wks.lb_pred_10[dates.>=Date(2022, 5, 1)]



vline!(
    plt_chng_oth,
    [Date(2022, 7, 23)],
    lw = 3,
    color = :black,
    ls = :dot,
    lab = "",
    annotation = (Date(2022, 7, 23) + Day(7), 0.21, "WHO declaration"),
    annotationrotation = 270,
)

display(plt_chng_oth)

CSV.write("projections/R_other" * string(wks[end]) * ".csv", R_oth_risk)

## Combined plot

plt = plot(
    plt_R_gbmsm,
    plt_chng_oth,
    size = (1600, 800),
    dpi = 250,
    left_margin = 10mm,
    bottom_margin = 10mm,
    right_margin = 10mm,
    layout = (1, 2),
)
display(plt)
savefig(plt, "plots/risk_over_time" * string(wks[end]) * ".png")



# plt = plot(
#     plt_chng,
#     plt_chng_oth,
#     plt_prev,
#     plt_prev_overall,
#     size=(1750, 1600),
#     dpi=250,
#     left_margin=10mm,
#     bottom_margin=10mm,
#     right_margin=10mm,
#     layout=(2, 2),
# )
# display(plt)
# savefig(plt, "plots/change_and_prevalence" * string(wks[end]) * ".png")


##Cumulative infections

d1, d2 = size(mpxv_wkly)

cuminf_cred_int = prev_cred_intervals([
    cumsum(pred[2], dims = 1) for pred in preds_and_incidence_interventions
])
cuminf_cred_int_4wkrev = prev_cred_intervals([
    cumsum(pred[2], dims = 1) for pred in preds_and_incidence_interventions_4wkrev
])
cuminf_cred_int_12wkrev = prev_cred_intervals([
    cumsum(pred[2], dims = 1) for pred in preds_and_incidence_interventions_12wkrev
])

cuminf_cred_int_cvac = prev_cred_intervals([
    cumsum(pred[2], dims = 1) for pred in preds_and_incidence_interventions_cvac
])
cuminf_cred_int_cvac_4wkrev = prev_cred_intervals([
    cumsum(pred[2], dims = 1) for pred in preds_and_incidence_interventions_cvac_4wkrev
])
cuminf_cred_int_cvac_12wkrev = prev_cred_intervals([
    cumsum(pred[2], dims = 1) for pred in preds_and_incidence_interventions_cvac_12wkrev
])

cuminf_cred_int_overall = prev_cred_intervals([
    cumsum(sum(pred[2][:, 1:10], dims = 2), dims = 1) for
    pred in preds_and_incidence_interventions
])
cuminf_cred_int_overall_4wkrev = prev_cred_intervals([
    cumsum(sum(pred[2][:, 1:10], dims = 2), dims = 1) for
    pred in preds_and_incidence_interventions_4wkrev
])

cuminf_cred_int_overall_12wkrev = prev_cred_intervals([
    cumsum(sum(pred[2][:, 1:10], dims = 2), dims = 1) for
    pred in preds_and_incidence_interventions_12wkrev
])

cuminf_cred_int_overall_cvac_4wkrev = prev_cred_intervals([
    cumsum(sum(pred[2][:, 1:10], dims = 2), dims = 1) for
    pred in preds_and_incidence_interventions_cvac_4wkrev
])


##

N_msm_grp = N_msm .* ps'
_wks = long_wks .- Day(7)


exposure = DataFrame()
date_exposure = Date(2022, 10, 1)

f = findfirst(_wks .>= date_exposure) #Date for plotting exposure

#Population sizes and generate xtick labels
pop_sizes = [(N_uk - N_msm); N_msm_grp[:]]
prop_inf = [
    cuminf_cred_int.mean_pred[f, 11] / (N_uk - N_msm)
    cuminf_cred_int.mean_pred[f, 1:10] ./ N_msm_grp[:]
]
100 * cuminf_cred_int_overall.mean_pred[f] / N_msm
100 * (cuminf_cred_int_overall.mean_pred[f] - cuminf_cred_int_overall.lb_pred_10[f]) / N_msm
100 * (cuminf_cred_int_overall.mean_pred[f] + cuminf_cred_int_overall.ub_pred_10[f]) / N_msm

mnthly_cnts =
    xs ./ 12 .|>
    x ->
        round(x, sigdigits = 2) .|>
        x -> string(x) .|> str -> str[(end-1):end] == ".0" ? str[1:(end-2)] : str
xtickstr = Vector{String}(undef, 10)
# xtickstr[1] = "< " * mnthly_cnts[1]
for k = 1:9
    xtickstr[k] = mnthly_cnts[k] * "-" * mnthly_cnts[k+1]
end
xtickstr[10] = "> " * mnthly_cnts[10]
xtickstr = ["non-GBMSM"; xtickstr]


plt_prop = bar(
    pop_sizes ./ N_uk,
    yscale = :log10,
    title = "Risk groups and proportions infected (1st Oct)",
    xticks = (1:11, xtickstr),
    yticks = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
    ylabel = "Proportion of population",
    xlabel = "Monthly new sexual partners",
    bar_width = 0.9,
    size = (1600, 600),
    dpi = 250,
    tickfont = 13,
    titlefont = 28,
    guidefont = 24,
    legendfont = 16,
    left_margin = 15mm,
    bottom_margin = 15mm,
    top_margin = 10mm,
    color = :blue,
    lab = "Proportion of group uninfected",
)

bar!(
    plt_prop,
    pop_sizes ./ N_uk,
    bar_width = prop_inf .* 0.9,
    yscale = :log10,
    lab = "Proportion of group infected",
    color = :red,
)

exposure[:, "Groups"] = xtickstr
exposure[:, "Proportion of population"] = pop_sizes ./ N_uk
exposure[:, "Proportion infected (post. mean)"] = prop_inf
exposure[:, "Proportion infected (post. 10%)"] = [
    (cuminf_cred_int.mean_pred[f, 11] - cuminf_cred_int.lb_pred_10[f, 11]) / (N_uk - N_msm)
    (cuminf_cred_int.mean_pred[f, 1:10] .- cuminf_cred_int.lb_pred_10[f, 1:10]) ./
    N_msm_grp[:]
]
exposure[:, "Proportion infected (post. 90%)"] = [
    (cuminf_cred_int.mean_pred[f, 11] + cuminf_cred_int.ub_pred_10[f, 11]) / (N_uk - N_msm)
    (cuminf_cred_int.mean_pred[f, 1:10] .+ cuminf_cred_int.ub_pred_10[f, 1:10]) ./
    N_msm_grp[:]
]

display(plt_prop)
CSV.write("projections/population_exposure_" * string(date_exposure) * ".csv", exposure)


## Main figure 1

layout = @layout [a b; c d]
fig1 = plot(
    plt_R_gbmsm,
    plt_chng_oth,
    plt_msm,
    plt_nmsm,
    # plt_prop,
    size = (1750, 1600),
    dpi = 250,
    left_margin = 10mm,
    bottom_margin = 10mm,
    right_margin = 10mm,
    top_margin = 5mm,
    layout = layout,
)

display(fig1)
savefig(fig1, "plots/main_figure1_" * string(wks[end]) * ".png")

## Main figure 2

layout = @layout [a; b c]
fig2 = plot(
    plt_prop,
    plt_cm_msm,
    plt_cm_nmsm,
    size = (1750, 1600),
    dpi = 250,
    left_margin = 10mm,
    bottom_margin = 10mm,
    right_margin = 10mm,
    top_margin = 5mm,
    layout = layout,
)

display(fig2)
savefig(fig2, "plots/main_figure2_" * string(wks[end]) * ".png")
savefig(fig2, "plots/main_figure2_" * string(wks[end]) * ".svg")


## Plot uncontrolled epidemic

case_projections_unmitigated = DataFrame()
case_projections_unmitigated[:, "Date"] = long_wks

plt_msm_unmit = plot(;
    ylabel = "Weekly cases",
    title = "UK Monkeypox Case Projection (GBMSM; unmitigated)",# yscale=:log10,
    legend = :topright,
    # yticks=([1, 2, 11, 101, 1001], [0, 1, 10, 100, 1000]),
    # ylims=(-5, 700),
    xticks = (
        [Date(2022, 5, 1) + Month(k) for k = 0:11],
        [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:11],
    ),
    left_margin = 5mm,
    size = (800, 600),
    dpi = 250,
    tickfont = 13,
    titlefont = 17,
    guidefont = 24,
    legendfont = 11,
)

plot!(
    plt_msm_unmit,
    long_wks,
    cred_int_unmitigated.mean_pred[:, 1],
    ribbon = (cred_int_unmitigated.lb_pred_10[:, 1], cred_int_unmitigated.ub_pred_10[:, 1]),
    lw = 3,
    color = :green,
    fillalpha = 0.2,
    lab = "",
)

scatter!(
    plt_msm_unmit,
    wks[3:end],
    mpxv_wkly[3:end, 1],
    lab = "Actual cases",
    ms = 6,
    color = :black,
)


case_projections_unmitigated[:, "Projected GBMSM cases (post. mean; unmitigated)"] =
    cred_int_unmitigated.mean_pred[:, 1]
case_projections_unmitigated[:, "Projected GBMSM cases (post. 10%; unmitigated)"] =
    cred_int_unmitigated.mean_pred[:, 1] .- cred_int_unmitigated.lb_pred_10[:, 1]
case_projections_unmitigated[:, "Projected GBMSM cases (post. 90%; unmitigated)"] =
    cred_int_unmitigated.mean_pred[:, 1] .+ cred_int_unmitigated.ub_pred_10[:, 1]

case_projections_unmitigated[:, "Projected GBMSM cum. cases (post. mean; unmitigated)"] =
    cred_int_cum_cases_unmitigated.mean_pred[:, 1]
case_projections_unmitigated[:, "Projected GBMSM cum. cases (post. 10%; unmitigated)"] =
    cred_int_cum_cases_unmitigated.mean_pred[:, 1] .-
    cred_int_cum_cases_unmitigated.lb_pred_10[:, 1]
case_projections_unmitigated[:, "Projected GBMSM cum. cases (post. 90%; unmitigated)"] =
    cred_int_cum_cases_unmitigated.mean_pred[:, 1] .+
    cred_int_cum_cases_unmitigated.ub_pred_10[:, 1]

display(plt_msm_unmit)

##
plt_nonmsm_unmit = plot(;
    ylabel = "Weekly cases",
    title = "UK Monkeypox Case Projection (non-GBMSM; unmitigated)",# yscale=:log10,
    legend = :topright,
    # yticks=([1, 2, 11, 101, 1001], [0, 1, 10, 100, 1000]),
    # ylims=(-5, 700),
    xticks = (
        [Date(2022, 5, 1) + Month(k) for k = 0:11],
        [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:11],
    ),
    left_margin = 5mm,
    size = (800, 600),
    dpi = 250,
    tickfont = 13,
    titlefont = 16,
    guidefont = 24,
    legendfont = 11,
)

plot!(
    plt_nonmsm_unmit,
    long_wks,
    cred_int_unmitigated.mean_pred[:, 2],
    ribbon = (cred_int_unmitigated.lb_pred_10[:, 2], cred_int_unmitigated.ub_pred_10[:, 2]),
    lw = 3,
    color = :green,
    fillalpha = 0.2,
    lab = "",
)

scatter!(
    plt_nonmsm_unmit,
    wks[3:end],
    mpxv_wkly[3:end, 2],
    lab = "Actual cases",
    ms = 6,
    color = :black,
)
display(plt_nonmsm_unmit)

case_projections_unmitigated[:, "Projected non-GBMSM cases (post. mean; unmitigated)"] =
    cred_int_unmitigated.mean_pred[:, 2]
case_projections_unmitigated[:, "Projected non-GBMSM cases (post. 10%; unmitigated)"] =
    cred_int_unmitigated.mean_pred[:, 2] .- cred_int_unmitigated.lb_pred_10[:, 2]
case_projections_unmitigated[:, "Projected non-GBMSM cases (post. 90%; unmitigated)"] =
    cred_int_unmitigated.mean_pred[:, 2] .+ cred_int_unmitigated.ub_pred_10[:, 2]

case_projections_unmitigated[
    :,
    "Projected non-GBMSM cum. cases (post. mean; unmitigated)",
] = cred_int_cum_cases_unmitigated.mean_pred[:, 2]
case_projections_unmitigated[:, "Projected non-GBMSM cum. cases (post. 10%; unmitigated)"] =
    cred_int_cum_cases_unmitigated.mean_pred[:, 2] .-
    cred_int_cum_cases_unmitigated.lb_pred_10[:, 2]
case_projections_unmitigated[:, "Projected non-GBMSM cum. cases (post. 90%; unmitigated)"] =
    cred_int_cum_cases_unmitigated.mean_pred[:, 2] .+
    cred_int_cum_cases_unmitigated.ub_pred_10[:, 2]

##

plt_unmit = plot(
    plt_msm_unmit,
    plt_nonmsm_unmit,
    size = (1750, 800),
    dpi = 250,
    left_margin = 10mm,
    bottom_margin = 10mm,
    right_margin = 10mm,
    layout = (1, 2),
)
savefig(plt_unmit, "plots/unmitigated_mpx.png")
CSV.write(
    "projections/case_projections_unmitigated_" * string(wks[end]) * ".csv",
    case_projections_unmitigated,
)
