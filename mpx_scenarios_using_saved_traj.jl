using Distributions, StatsBase, StatsPlots
using Plots.PlotMeasures, CSV, DataFrames
using LinearAlgebra, RecursiveArrayTools, Dates
using ProgressMeter
using OrdinaryDiffEq, ApproxBayes, MCMCChains
using JLD2
using MonkeypoxUK

## MSM data with data inference

past_mpxv_data_inferred = CSV.File("data/weekly_data_imputation_2022-09-30.csv",
                                missingstring = "NA") |> DataFrame

wks = Date.(past_mpxv_data_inferred.week, DateFormat("dd/mm/yyyy"))

include("setup_model.jl");

## 

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

wks = Date.(past_mpxv_data_inferred.week[1:size(mpxv_wkly, 1)], DateFormat("dd/mm/yyyy"))
ts = wks .|> d -> d - Date(2021, 12, 31) .|> t -> t.value

wkly_vaccinations = [
    [zeros(12); 1000; 2000; fill(5000, 4)] * 1.675
    fill(650, 18)
]
print("cum. vacs = $(sum(wkly_vaccinations))")

## Load posterior draws and saved trajectiories

date_str = "2022-09-26"
# description_str = "no_bv_cng"
description_str = "no_ngbmsm_chg"
# description_str = ""

param_draws = load("posteriors/posterior_param_draws_" * date_str * description_str * ".jld2")["param_draws"]
detected_cases = load("posteriors/posterior_detected_cases_" * date_str * description_str * ".jld2")["detected_cases"]
vac_effs = load("posteriors/posterior_vac_effectivenesses_" * date_str * description_str * ".jld2")["vac_effectivenesses"]
no_vac_states = load("posteriors/posterior_begin_vac_states_" * date_str * description_str * ".jld2")["begin_vac_states"]
sept_states = load("posteriors/posterior_begin_sept_states_" * date_str * description_str * ".jld2")["begin_sept_states"]
end_states = load("posteriors/posterior_end_states_" * date_str * description_str * ".jld2")["end_states"]
incidences = load("posteriors/posterior_incidences_" * date_str * description_str * ".jld2")["incidences"]

##

param_draws = [θ[1:end] |> x -> vcat(x,0.0) for θ in param_draws]


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
# include("mpx_projections_dev.jl")
## Define counterfactuals and projections

d1, d2 = size(mpxv_wkly)
n_samples = 2000
n_wks_lookahead = 26
f_rev = findfirst(wks .> Date(2022,9,1))
f_novac = findfirst(wks .== Date(2022,7,25))

wks_proj_fromend = [wks[end] + Day(7k) for k = 1:n_wks_lookahead]
# wks_proj_fromend = [wks[8] + Day(7k) for k = 1:n_wks_lookahead]
wks_reversion = [wks[f_rev] + Day(7k) for k = 1:n_lookaheadweeks + 3]
wks_no_vacs = [wks[f_novac] + Day(7k) for k = 1:n_lookaheadweeks + 9]

# Checks
println("Projection from end and projection from first week in september have same end data is ", wks_proj_fromend[end] == wks_reversion[end])
println("Projection from end and projection from last week with no vaccines have same end data is ", wks_proj_fromend[end] == wks_no_vacs[end])

# Make a set of projections

proj_fromend = [(ts = wks_proj_fromend .|> d -> (d - Date(2021,12,31)).value |> Float64,
                        wkly_vaccinations = wkly_vaccinations[(d1+1):end],
                        vac_effectiveness = vac_effs[k]) for k = 1:n_samples]

proj_reversion = [(ts = wks_reversion .|> d -> (d - Date(2021,12,31)).value |> Float64,
                        wkly_vaccinations = wkly_vaccinations[((d1-3)+1):end],
                        vac_effectiveness = vac_effs[k]) for k = 1:n_samples]

proj_no_vaccines = [(ts = wks_no_vacs .|> d -> (d - Date(2021,12,31)).value |> Float64,
                        wkly_vaccinations = zeros(100),
                        vac_effectiveness = vac_effs[k]) for k = 1:n_samples]

projections_from_end = @showprogress 0.1 "MPX Forecasts:" map((θ, interventions, state) ->  mpx_sim_function_projections(θ, constants, interventions, state),
                    param_draws,
                    proj_fromend,
                    end_states)  

projections_reversions_4wk_rev = @showprogress 0.1 "MPX 4 week reversion:" map((θ, interventions, state) ->  mpx_sim_function_projections(θ, constants, interventions, state, 4),
                    param_draws,
                    proj_reversion,
                    sept_states)                      

projections_reversions_12wk_rev = @showprogress 0.1 "MPX 12 week reversion:" map((θ, interventions, state) ->  mpx_sim_function_projections(θ, constants, interventions, state, 12),
                    param_draws,
                    proj_reversion,
                    sept_states)    

projections_reversions_4wk_rev_no_vac = @showprogress 0.1 "MPX 4 week reversion (no vaccines):" map((θ, interventions, state) ->  mpx_sim_function_projections(θ, constants, interventions, state, 4),
                    param_draws,
                    proj_no_vaccines,
                    no_vac_states)                      

projections_reversions_12wk_rev_no_vac = @showprogress 0.1 "MPX 12 week reversion (no vaccines):" map((θ, interventions, state) ->  mpx_sim_function_projections(θ, constants, interventions, state, 12),
                    param_draws,
                    proj_no_vaccines,
                    no_vac_states)                        


## 

cred_proj = MonkeypoxUK.cred_intervals([proj.detected_cases for proj in projections_from_end],central_est = :median)

cred_int = MonkeypoxUK.cred_intervals(detected_cases, central_est = :median)
cred_int_12wk = MonkeypoxUK.cred_intervals([proj.detected_cases for proj in projections_reversions_12wk_rev],central_est = :median)
cred_int_12wk_no_vacs = MonkeypoxUK.cred_intervals([proj.detected_cases for proj in projections_reversions_12wk_rev_no_vac], central_est = :median)
cred_int_4wk = MonkeypoxUK.cred_intervals([proj.detected_cases for proj in projections_reversions_4wk_rev], central_est = :median)
cred_int_4wk_no_vacs = MonkeypoxUK.cred_intervals([proj.detected_cases for proj in projections_reversions_4wk_rev_no_vac], central_est = :median)

cum_cred_int = MonkeypoxUK.cred_intervals(cumsum(detected_cases, dims = 1), central_est = :median)
cum_cred_int_12wk = MonkeypoxUK.cred_intervals([proj.detected_cases for proj in projections_reversions_12wk_rev] .|> x -> cumsum(x, dims = 1),central_est = :median)
cum_cred_int_12wk_no_vacs = MonkeypoxUK.cred_intervals([proj.detected_cases for proj in projections_reversions_12wk_rev_no_vac] .|> x -> cumsum(x, dims = 1), central_est = :median)
cum_cred_int_4wk = MonkeypoxUK.cred_intervals([proj.detected_cases for proj in projections_reversions_4wk_rev] .|> x -> cumsum(x, dims = 1), central_est = :median)
cum_cred_int_4wk_no_vacs = MonkeypoxUK.cred_intervals([proj.detected_cases for proj in projections_reversions_4wk_rev_no_vac] .|> x -> cumsum(x, dims = 1), central_est = :median)

cum_cred_infs = MonkeypoxUK.matrix_cred_intervals(cumsum.(incidences, dims = 1), central_est = :median)



## MSM projections

plt_msm = plot(;
    ylabel = "Weekly cases",
    title = "UK Monkeypox Case Projections (GBMSM)",
    legend = :topright,
    # ylims = (-5, 700),
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
    wks[3:size(cred_int.median_pred,1)+2],
    cred_int.median_pred[:, 1],
    ribbon = (cred_int.lb_pred_25[:, 1], cred_int.ub_pred_25[:, 1]),
    lw = 3,
    color = :black,
    fillalpha = 0.15,
    lab = "Fitted trajectory",
)

plot!(
    plt_msm,
    wks[3:size(cred_int.median_pred,1)+2],
    cred_int.median_pred[:, 1],
    ribbon = (cred_int.lb_pred_10[:, 1], cred_int.ub_pred_10[:, 1]),
    lw = 0,
    color = :black,
    fillalpha = 0.15,
    lab = "",
)

plot!(
    plt_msm,
    wks[3:size(cred_int.median_pred,1)+2],
    cred_int.median_pred[:, 1],
    ribbon = (cred_int.lb_pred_025[:, 1], cred_int.ub_pred_025[:, 1]),
    lw = 0,
    color = :black,
    fillalpha = 0.15,
    lab = "",
)

# plot!(plt_msm,
#         wks_proj_fromend,
#         cred_proj.median_pred[:,1],
#         ribbon = (cred_proj.lb_pred_025[:, 1], cred_proj.ub_pred_025[:, 1]),
#         lab = "bev. change")

plot!(
    plt_msm,
    wks_reversion,
    cred_int_12wk.median_pred[:, 1],
    ribbon = (cred_int_12wk.lb_pred_25[:, 1], cred_int_12wk.ub_pred_25[:, 1]),
    lw = 3,
    color = :blue,
    fillalpha = 0.15,
    lab = "12 week reversion",
)

plot!(
    plt_msm,
    wks_reversion,
    cred_int_12wk.median_pred[:, 1],
    ribbon = (cred_int_12wk.lb_pred_10[:, 1], cred_int_12wk.ub_pred_10[:, 1]),
    lw = 0,
    color = :blue,
    fillalpha = 0.15,
    lab = "",
)

plot!(
    plt_msm,
    wks_reversion,
    cred_int_12wk.median_pred[:, 1],
    ribbon = (cred_int_12wk.lb_pred_025[:, 1], cred_int_12wk.ub_pred_025[:, 1]),
    lw = 0,
    color = :blue,
    fillalpha = 0.15,
    lab = "",
)

plot!(
    plt_msm,
    wks_no_vacs,
    cred_int_12wk_no_vacs.median_pred[:, 1],
    # ribbon = (cred_int_12wk.lb_pred_25[:, 1], cred_int_12wk.ub_pred_25[:, 1]),
    lw = 3,
    color = :blue,
    ls = :dash,
    fillalpha = 0.15,
    lab = "12 week reversion (no vaccines)",
)



plot!(
    plt_msm,
    wks_reversion,
    cred_int_4wk.median_pred[:, 1],
    ribbon = (cred_int_4wk.lb_pred_25[:, 1], cred_int_4wk.ub_pred_25[:, 1]),
    lw = 3,
    color = :red,
    fillalpha = 0.15,
    lab = "4 week reversion",
)

plot!(
    plt_msm,
    wks_reversion,
    cred_int_4wk.median_pred[:, 1],
    ribbon = (cred_int_4wk.lb_pred_10[:, 1], cred_int_4wk.ub_pred_10[:, 1]),
    lw = 0,
    color = :red,
    fillalpha = 0.15,
    lab = "",
)

plot!(
    plt_msm,
    wks_reversion,
    cred_int_4wk.median_pred[:, 1],
    ribbon = (cred_int_4wk.lb_pred_025[:, 1], cred_int_4wk.ub_pred_025[:, 1]),
    lw = 0,
    color = :red,
    fillalpha = 0.15,
    lab = "",
)

plot!(
    plt_msm,
    wks_no_vacs,
    cred_int_4wk_no_vacs.median_pred[:, 1],
    lw = 3,
    color = :red,
    ls = :dash,
    fillalpha = 0.15,
    lab = "4 week reversion (no vaccines)",
)

scatter!(
    plt_msm,
    wks[[1,2,size(wks,1)]],
    mpxv_wkly[[1,2,size(wks,1)], 1],
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


# CSV.write("projections/gbmsm_case_projections" * string(wks[end]) * ".csv", gbmsm_case_projections)
display(plt_msm)

##

plt_nmsm = plot(;
    ylabel = "Weekly cases",
    title = "UK Monkeypox Case Projections (non-GBMSM)",
    legend = :topright,
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
    wks[3:end],
    cred_int.median_pred[:, 2],
    ribbon = (cred_int.lb_pred_25[:, 2], cred_int.ub_pred_25[:, 2]),
    lw = 3,
    color = :black,
    fillalpha = 0.2,
    lab = "Fitted trajectory",
)

plot!(
    plt_nmsm,
    wks[3:end],
    cred_int.median_pred[:, 2],
    ribbon = (cred_int.lb_pred_10[:, 2], cred_int.ub_pred_10[:, 2]),
    lw = 0,
    color = :black,
    fillalpha = 0.2,
    lab = "",
)

plot!(
    plt_nmsm,
    wks[3:end],
    cred_int.median_pred[:, 2],
    ribbon = (cred_int.lb_pred_025[:, 2], cred_int.ub_pred_025[:, 2]),
    lw = 0,
    color = :black,
    fillalpha = 0.2,
    lab = "",
)

plot!(
    plt_nmsm,
    wks_reversion,
    cred_int_12wk.median_pred[:, 2],
    ribbon = (cred_int_12wk.lb_pred_25[:, 2], cred_int_12wk.ub_pred_25[:, 2]),
    lw = 3,
    color = :blue,
    fillalpha = 0.2,
    lab = "12 week reversion",
)

plot!(
    plt_nmsm,
    wks_reversion,
    cred_int_12wk.median_pred[:, 2],
    ribbon = (cred_int_12wk.lb_pred_10[:, 2], cred_int_12wk.ub_pred_10[:, 2]),
    lw = 0,
    color = :blue,
    fillalpha = 0.2,
    lab = "",
)

plot!(
    plt_nmsm,
    wks_reversion,
    cred_int_12wk.median_pred[:, 2],
    ribbon = (cred_int_12wk.lb_pred_025[:, 2], cred_int_12wk.ub_pred_025[:, 2]),
    lw = 0,
    color = :blue,
    fillalpha = 0.2,
    lab = "",
)

plot!(
    plt_nmsm,
    wks_no_vacs,
    cred_int_12wk_no_vacs.median_pred[:, 2],
    # ribbon = (cred_int_12wk.lb_pred_25[:, 2], cred_int_12wk.ub_pred_25[:, 2]),
    ls = :dash,
    lw = 3,
    color = :blue,
    fillalpha = 0.2,
    lab = "12 week reversion (no vaccines)",
)

plot!(
    plt_nmsm,
    wks_reversion,
    cred_int_4wk.median_pred[:, 2],
    ribbon = (cred_int_4wk.lb_pred_25[:, 2], cred_int_4wk.ub_pred_25[:, 2]),
    lw = 3,
    color = :red,
    fillalpha = 0.2,
    lab = "4 week reversion",
)

plot!(
    plt_nmsm,
    wks_reversion,
    cred_int_4wk.median_pred[:, 2],
    ribbon = (cred_int_4wk.lb_pred_10[:, 2], cred_int_4wk.ub_pred_10[:, 2]),
    lw = 0,
    color = :red,
    fillalpha = 0.2,
    lab = "",
)

plot!(
    plt_nmsm,
    wks_reversion,
    cred_int_4wk.median_pred[:, 2],
    ribbon = (cred_int_4wk.lb_pred_025[:, 2], cred_int_4wk.ub_pred_025[:, 2]),
    lw = 0,
    color = :red,
    fillalpha = 0.2,
    lab = "",
)

plot!(
    plt_nmsm,
    wks_no_vacs,
    cred_int_4wk_no_vacs.median_pred[:, 2],
    # ribbon = (cred_int_12wk.lb_pred_25[:, 2], cred_int_12wk.ub_pred_25[:, 2]),
    ls = :dash,
    lw = 3,
    color = :red,
    fillalpha = 0.2,
    lab = "4 week reversion (no vaccines)",
)

scatter!(
    plt_nmsm,
    wks[[1,2,size(wks,1)]],
    mpxv_wkly[[1,2,size(wks,1)], 2],
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

display(plt_nmsm)

## Change in transmission over time

long_wks_reversion = [wks[1:(end-3)]; wks_reversion]
ts_reversion = (long_wks_reversion[1] - Date(2021,12,31)).value:1.0:(long_wks_reversion[end] - Date(2021,12,31)).value

transmission_risks_with_vaccines_no_rev = map(θ -> reproductive_ratios(θ, constants, ts_reversion),param_draws)
transmission_risks_with_vaccines_12wk_rev = map(θ -> reproductive_ratios(θ, constants, 12, ts_reversion),param_draws)
transmission_risks_with_vaccines_4wk_rev = map(θ -> reproductive_ratios(θ, constants, 4, ts_reversion),param_draws)

creds_R0_gbmsm_no_rev =  matrix_cred_intervals([reshape(R.R₀_gbmsm, length(R.R₀_gbmsm), 1) for R in transmission_risks_with_vaccines_no_rev], central_est = :median)
creds_R0_ngbmsm_no_rev =  matrix_cred_intervals([reshape(R.R₀_ngbmsm, length(R.R₀_gbmsm), 1) for R in transmission_risks_with_vaccines_no_rev], central_est = :median)
creds_R0_gbmsm_12wk =  matrix_cred_intervals([reshape(R.R₀_gbmsm, length(R.R₀_gbmsm), 1) for R in transmission_risks_with_vaccines_12wk_rev], central_est = :median)
creds_R0_ngbmsm_12wk =  matrix_cred_intervals([reshape(R.R₀_ngbmsm, length(R.R₀_gbmsm), 1) for R in transmission_risks_with_vaccines_12wk_rev], central_est = :median)
creds_R0_gbmsm_4wk =  matrix_cred_intervals([reshape(R.R₀_gbmsm, length(R.R₀_gbmsm), 1) for R in transmission_risks_with_vaccines_4wk_rev], central_est = :median)
creds_R0_ngbmsm_4wk =  matrix_cred_intervals([reshape(R.R₀_ngbmsm, length(R.R₀_gbmsm), 1) for R in transmission_risks_with_vaccines_4wk_rev], central_est = :median)

##

dates = [Date(2021, 12, 31) + Day(t) for t in ts_reversion]
f = findfirst(dates .== Date(2022, 7, 23))
f2 = findfirst(dates .== Date(2022, 9, 1))
f3 = findfirst(dates .== wks[end-1])

# Posterior probability of >10% decrease in risk
# p_sx_risk_pheic = mean([p_trans[f] < p_trans[1] * 0.9 for p_trans in p_sx_trans_risks])

plt_R_gbmsm = plot(
    dates[1:f3],
    creds_R0_gbmsm_no_rev.median_pred[1:f3],
    ribbon = (creds_R0_gbmsm_no_rev.lb_pred_25[1:f3], creds_R0_gbmsm_no_rev.ub_pred_25[1:f3]),
    lw = 3,
    fillalpha = 0.2,
    lab = "Fitted",
    title = "Reproductive number (GBMSM)",
    ylabel = "R₀(t) (GBMSM)",
    xlims = (long_wks[1] - Day(7), long_wks[end] + Day(7)),
    ylims = (0, 10.5),
    xticks = (
        [Date(2022, 5, 1) + Month(k) for k = 0:11],
        [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:11],
    ),
    yticks = 0:10,
    color = :black,
    left_margin = 5mm,
    size = (800, 600),
    dpi = 250,
    tickfont = 13,
    titlefont = 22,
    guidefont = 18,
    legendfont = 11,
)

plot!(plt_R_gbmsm,
        dates[1:f3],
        creds_R0_gbmsm_no_rev.median_pred[1:f3],
        ribbon = (creds_R0_gbmsm_no_rev.lb_pred_10[1:f3], creds_R0_gbmsm_no_rev.ub_pred_10[1:f3]),
        fillalpha = 0.2,
        lab = "",
        lw = 0,
        c = :black,
        )

plot!(plt_R_gbmsm,
        dates[1:f3],
        creds_R0_gbmsm_no_rev.median_pred[1:f3],
        ribbon = (creds_R0_gbmsm_no_rev.lb_pred_025[1:f3], creds_R0_gbmsm_no_rev.ub_pred_025[1:f3]),
        fillalpha = 0.2,
        lab = "",
        lw = 0,
        c = :black,
        )
 
plot!(plt_R_gbmsm,
        dates[f2:end],
        creds_R0_gbmsm_4wk.median_pred[f2:end],
        ribbon = (creds_R0_gbmsm_4wk.lb_pred_25[f2:end], creds_R0_gbmsm_4wk.ub_pred_25[f2:end]),
        fillalpha = 0.2,
        lab = "4 weeks reversion",
        lw = 3,
        c = :red,
        )

plot!(plt_R_gbmsm,
        dates[f2:end],
        creds_R0_gbmsm_4wk.median_pred[f2:end],
        ribbon = (creds_R0_gbmsm_4wk.lb_pred_10[f2:end], creds_R0_gbmsm_4wk.ub_pred_10[f2:end]),
        fillalpha = 0.2,
        lab = "",
        lw = 3,
        c = :red,
        )        

plot!(plt_R_gbmsm,
        dates[f2:end],
        creds_R0_gbmsm_4wk.median_pred[f2:end],
        ribbon = (creds_R0_gbmsm_4wk.lb_pred_025[f2:end], creds_R0_gbmsm_4wk.ub_pred_025[f2:end]),
        fillalpha = 0.2,
        lab = "",
        lw = 3,
        c = :red,
        )  

plot!(plt_R_gbmsm,
        dates[f2:end],
        creds_R0_gbmsm_12wk.median_pred[f2:end],
        ribbon = (creds_R0_gbmsm_12wk.lb_pred_25[f2:end], creds_R0_gbmsm_12wk.ub_pred_25[f2:end]),
        fillalpha = 0.2,
        lab = "12 week reversion",
        lw = 3,
        c = :blue,
        )

plot!(plt_R_gbmsm,
        dates[f2:end],
        creds_R0_gbmsm_12wk.median_pred[f2:end],
        ribbon = (creds_R0_gbmsm_12wk.lb_pred_10[f2:end], creds_R0_gbmsm_12wk.ub_pred_10[f2:end]),
        fillalpha = 0.2,
        lab = "",
        lw = 3,
        c = :blue,
        )


plot!(plt_R_gbmsm,
        dates[f2:end],
        creds_R0_gbmsm_12wk.median_pred[f2:end],
        ribbon = (creds_R0_gbmsm_12wk.lb_pred_025[f2:end], creds_R0_gbmsm_12wk.ub_pred_025[f2:end]),
        fillalpha = 0.2,
        lab = "",
        lw = 3,
        c = :blue,
        )        
        
plot!(plt_R_gbmsm,
        dates[210:end],
        creds_R0_gbmsm_4wk.median_pred[210:end],
        alpha = 0.75,
        lab = "",
        ls = :dash,
        lw = 3,
        c = :red,
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

##
dates = [Date(2021, 12, 31) + Day(t) for t in ts_reversion]
f = findfirst(dates .== Date(2022, 7, 23))
f2 = findfirst(dates .== Date(2022, 9, 1))
f3 = findfirst(dates .== wks[end-1])


plt_R_ngbmsm = plot(
    dates[1:f3],
    creds_R0_ngbmsm_no_rev.median_pred[1:f3],
    ribbon = (
        creds_R0_ngbmsm_no_rev.lb_pred_25[1:f3],
        creds_R0_ngbmsm_no_rev.ub_pred_25[1:f3],
    ),
    lw = 3,
    fillalpha = 0.2,
    lab = "Fitted",
    title = "Reproductive number (other contacts)",
    ylabel = "R₀(t) (other contacts)",
    xlims = (long_wks[1] - Day(7), long_wks[end] + Day(7)),
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


plot!(plt_R_ngbmsm,
        dates[1:f3],
        creds_R0_ngbmsm_no_rev.median_pred[1:f3],
        ribbon = (creds_R0_ngbmsm_no_rev.lb_pred_10[1:f3], creds_R0_ngbmsm_no_rev.ub_pred_10[1:f3]),
        fillalpha = 0.2,
        lab = "",
        lw = 0,
        c = :black,
        )

plot!(plt_R_ngbmsm,
        dates[1:f3],
        creds_R0_ngbmsm_no_rev.median_pred[1:f3],
        ribbon = (creds_R0_ngbmsm_no_rev.lb_pred_025[1:f3], creds_R0_ngbmsm_no_rev.ub_pred_025[1:f3]),
        fillalpha = 0.2,
        lab = "",
        lw = 0,
        c = :black,
        )
 
plot!(plt_R_ngbmsm,
        dates[f2:end],
        creds_R0_ngbmsm_4wk.median_pred[f2:end],
        ribbon = (creds_R0_ngbmsm_4wk.lb_pred_25[f2:end], creds_R0_ngbmsm_4wk.ub_pred_25[f2:end]),
        fillalpha = 0.2,
        lab = "4 weeks reversion",
        lw = 3,
        c = :red,
        )

plot!(plt_R_ngbmsm,
        dates[f2:end],
        creds_R0_ngbmsm_4wk.median_pred[f2:end],
        ribbon = (creds_R0_ngbmsm_4wk.lb_pred_10[f2:end], creds_R0_ngbmsm_4wk.ub_pred_10[f2:end]),
        fillalpha = 0.2,
        lab = "",
        lw = 3,
        c = :red,
        )        

plot!(plt_R_ngbmsm,
        dates[f2:end],
        creds_R0_ngbmsm_4wk.median_pred[f2:end],
        ribbon = (creds_R0_ngbmsm_4wk.lb_pred_025[f2:end], creds_R0_ngbmsm_4wk.ub_pred_025[f2:end]),
        fillalpha = 0.2,
        lab = "",
        lw = 3,
        c = :red,
        )  

plot!(plt_R_ngbmsm,
        dates[f2:end],
        creds_R0_ngbmsm_12wk.median_pred[f2:end],
        ribbon = (creds_R0_ngbmsm_12wk.lb_pred_25[f2:end], creds_R0_ngbmsm_12wk.ub_pred_25[f2:end]),
        fillalpha = 0.2,
        lab = "12 week reversion",
        lw = 3,
        c = :blue,
        )

plot!(plt_R_ngbmsm,
        dates[f2:end],
        creds_R0_ngbmsm_12wk.median_pred[f2:end],
        ribbon = (creds_R0_ngbmsm_12wk.lb_pred_10[f2:end], creds_R0_ngbmsm_12wk.ub_pred_10[f2:end]),
        fillalpha = 0.2,
        lab = "",
        lw = 3,
        c = :blue,
        )


plot!(plt_R_ngbmsm,
        dates[f2:end],
        creds_R0_ngbmsm_12wk.median_pred[f2:end],
        ribbon = (creds_R0_ngbmsm_12wk.lb_pred_025[f2:end], creds_R0_ngbmsm_12wk.ub_pred_025[f2:end]),
        fillalpha = 0.2,
        lab = "",
        lw = 3,
        c = :blue,
        )        
        
plot!(plt_R_ngbmsm,
        dates[210:end],
        creds_R0_ngbmsm_4wk.median_pred[210:end],
        alpha = 0.75,
        lab = "",
        ls = :dash,
        lw = 3,
        c = :red,
        )

        
vline!(
    plt_R_ngbmsm,
    [Date(2022, 7, 23)],
    lw = 3,
    color = :black,
    ls = :dot,
    lab = "",
    annotation = (Date(2022, 7, 23) + Day(7), 0.1, "WHO declaration"),
    annotationrotation = 270,
)     


## Combined plot

plt = plot(
    plt_R_gbmsm,
    plt_R_ngbmsm,
    size = (1600, 800),
    dpi = 250,
    left_margin = 10mm,
    bottom_margin = 10mm,
    right_margin = 10mm,
    layout = (1, 2),
)
display(plt)
savefig(plt, "plots/risk_over_time" * string(wks[end]) * ".png")


##Cumulative infections
# f1 = findfirst

cum_mpxv_cases = cumsum(mpxv_wkly,dims = 1)

plt_cm_msm = plot(;
    ylabel="Cumulative cases",
    title="UK Monkeypox cumulative case projections (GBMSM)",
    legend=:topleft,
    xticks=(
        [Date(2022, 5, 1) + Month(k) for k = 0:11],
        [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:11],
    ),
    left_margin=5mm,
    size=(800, 600),
    dpi=250,
    tickfont=15,
    titlefont=16,
    guidefont=24,
    legendfont=11
)


plot!(
    plt_cm_msm,
    wks_reversion,
    cum_mpxv_cases[d1-2,1] .+ cum_cred_int_4wk.median_pred[:, 1],
    ribbon=(
        cum_cred_int_4wk.lb_pred_25[:, 1],
        cum_cred_int_4wk.ub_pred_25[:, 1],
    ),
    lw=3,
    color=:red,
    fillalpha=0.2,
    lab="4 week reversion",
)

plot!(
    plt_cm_msm,
    wks_reversion,
    cum_mpxv_cases[d1-2,1] .+ cum_cred_int_4wk.median_pred[:, 1],
    ribbon=(
        cum_cred_int_4wk.lb_pred_10[:, 1],
        cum_cred_int_4wk.ub_pred_10[:, 1],
    ),
    lw=3,
    color=:red,
    fillalpha=0.2,
    lab="",
)

plot!(
    plt_cm_msm,
    wks_reversion,
    cum_mpxv_cases[d1-2,1] .+ cum_cred_int_4wk.median_pred[:, 1],
    ribbon=(
        cum_cred_int_4wk.lb_pred_025[:, 1],
        cum_cred_int_4wk.ub_pred_025[:, 1],
    ),
    lw=3,
    color=:red,
    fillalpha=0.2,
    lab="",
)

plot!(
    plt_cm_msm,
    wks_no_vacs,
    cum_mpxv_cases[d1-9,1] .+ cum_cred_int_4wk_no_vacs.median_pred[:, 1],
    ribbon=(
        cum_cred_int_4wk_no_vacs.lb_pred_25[:, 1],
        cum_cred_int_4wk_no_vacs.ub_pred_25[:, 1],
    ),
    lw=3,
    ls = :dash,
    color=:red,
    fillalpha=0.2,
    lab="4 week reversion (no vaccines)",
)

plot!(
    plt_cm_msm,
    wks_reversion,
    cum_mpxv_cases[d1-2,1] .+ cum_cred_int_12wk.median_pred[:, 1],
    ribbon=(
        cum_cred_int_12wk.lb_pred_25[:, 1],
        cum_cred_int_12wk.ub_pred_25[:, 1],
    ),
    lw=3,
    color=:blue,
    fillalpha=0.2,
    lab="12 week reversion",
)

plot!(
    plt_cm_msm,
    wks_reversion,
    cum_mpxv_cases[d1-2,1] .+ cum_cred_int_12wk.median_pred[:, 1],
    ribbon=(
        cum_cred_int_12wk.lb_pred_10[:, 1],
        cum_cred_int_12wk.ub_pred_10[:, 1],
    ),
    lw=0,
    color=:blue,
    fillalpha=0.2,
    lab="",
)

plot!(
    plt_cm_msm,
    wks_reversion,
    cum_mpxv_cases[d1-2,1] .+ cum_cred_int_12wk.median_pred[:, 1],
    ribbon=(
        cum_cred_int_12wk.lb_pred_025[:, 1],
        cum_cred_int_12wk.ub_pred_025[:, 1],
    ),
    lw=0,
    color=:blue,
    fillalpha=0.2,
    lab="",
)

plot!(
    plt_cm_msm,
    wks_no_vacs,
    cum_mpxv_cases[d1-9,1] .+ cum_cred_int_12wk_no_vacs.median_pred[:, 1],
    ribbon=(
        cum_cred_int_12wk_no_vacs.lb_pred_25[:, 1],
        cum_cred_int_12wk_no_vacs.ub_pred_25[:, 1],
    ),
    lw=3,
    ls = :dash,
    color=:blue,
    fillalpha=0.2,
    lab="12 week reversion (no vaccines)",
)


scatter!(
    plt_cm_msm,
    wks,
    cumsum(mpxv_wkly[:, 1], dims=1),
    lab="Data",
    ms=6,
    color=:black,
)

##

plt_cm_nmsm = plot(;
    ylabel="Cumulative cases",
    title="UK Monkeypox cumulative case projections (non-GBMSM)",
    legend=:topleft,
    xticks=(
        [Date(2022, 5, 1) + Month(k) for k = 0:11],
        [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:11],
    ),
    left_margin=5mm,
    size=(800, 600),
    dpi=250,
    tickfont=15,
    titlefont=16,
    guidefont=24,
    legendfont=11
)


plot!(
    plt_cm_nmsm,
    wks_reversion,
    cum_mpxv_cases[d1-2,2] .+ cum_cred_int_4wk.median_pred[:, 2],
    ribbon=(
        cum_cred_int_4wk.lb_pred_25[:, 2],
        cum_cred_int_4wk.ub_pred_25[:, 2],
    ),
    lw=3,
    color=:red,
    fillalpha=0.2,
    lab="4 week reversion",
)

plot!(
    plt_cm_nmsm,
    wks_reversion,
    cum_mpxv_cases[d1-2,2] .+ cum_cred_int_4wk.median_pred[:, 2],
    ribbon=(
        cum_cred_int_4wk.lb_pred_10[:, 2],
        cum_cred_int_4wk.ub_pred_10[:, 2],
    ),
    lw=3,
    color=:red,
    fillalpha=0.2,
    lab="",
)

plot!(
    plt_cm_nmsm,
    wks_reversion,
    cum_mpxv_cases[d1-2,2] .+ cum_cred_int_4wk.median_pred[:, 2],
    ribbon=(
        cum_cred_int_4wk.lb_pred_025[:, 2],
        cum_cred_int_4wk.ub_pred_025[:, 2],
    ),
    lw=3,
    color=:red,
    fillalpha=0.2,
    lab="",
)

plot!(
    plt_cm_nmsm,
    wks_no_vacs,
    cum_mpxv_cases[d1-9,2] .+ cum_cred_int_4wk_no_vacs.median_pred[:, 2],
    ribbon=(
        cum_cred_int_4wk_no_vacs.lb_pred_25[:, 2],
        cum_cred_int_4wk_no_vacs.ub_pred_25[:, 2],
    ),
    lw=3,
    ls = :dash,
    color=:red,
    fillalpha=0.2,
    lab="4 week reversion (no vaccines)",
)

plot!(
    plt_cm_nmsm,
    wks_reversion,
    cum_mpxv_cases[d1-2,2] .+ cum_cred_int_12wk.median_pred[:, 2],
    ribbon=(
        cum_cred_int_12wk.lb_pred_25[:, 2],
        cum_cred_int_12wk.ub_pred_25[:, 2],
    ),
    lw=3,
    color=:blue,
    fillalpha=0.2,
    lab="12 week reversion",
)

plot!(
    plt_cm_nmsm,
    wks_reversion,
    cum_mpxv_cases[d1-2,2] .+ cum_cred_int_12wk.median_pred[:, 2],
    ribbon=(
        cum_cred_int_12wk.lb_pred_10[:, 2],
        cum_cred_int_12wk.ub_pred_10[:, 2],
    ),
    lw=0,
    color=:blue,
    fillalpha=0.2,
    lab="",
)

plot!(
    plt_cm_nmsm,
    wks_reversion,
    cum_mpxv_cases[d1-2,2] .+ cum_cred_int_12wk.median_pred[:, 2],
    ribbon=(
        cum_cred_int_12wk.lb_pred_025[:, 2],
        cum_cred_int_12wk.ub_pred_025[:, 2],
    ),
    lw=0,
    color=:blue,
    fillalpha=0.2,
    lab="",
)

plot!(
    plt_cm_nmsm,
    wks_no_vacs,
    cum_mpxv_cases[d1-9,2] .+ cum_cred_int_12wk_no_vacs.median_pred[:, 2],
    ribbon=(
        cum_cred_int_12wk_no_vacs.lb_pred_25[:, 2],
        cum_cred_int_12wk_no_vacs.ub_pred_25[:, 2],
    ),
    lw=3,
    ls = :dash,
    color=:blue,
    fillalpha=0.2,
    lab="12 week reversion (no vaccines)",
)


scatter!(
    plt_cm_nmsm,
    wks,
    cumsum(mpxv_wkly[:, 2], dims=1),
    lab="Data",
    ms=6,
    color=:black,
)

##

N_msm_grp = N_msm .* ps'
_wks = long_wks .- Day(7)

date_exposure = Date(2022, 10, 1)

f = findfirst(_wks .>= date_exposure) #Date for plotting exposure

#Population sizes and generate xtick labels
pop_sizes = [(N_uk - N_msm); N_msm_grp[:]]
prop_inf = [
    cum_cred_infs.median_pred[end, 11] / (N_uk - N_msm)
    cum_cred_infs.median_pred[end, 1:10] ./ N_msm_grp[:]
]
# 100 * cuminf_cred_int_overall.mean_pred[f] / N_msm
# 100 * (cuminf_cred_int_overall.mean_pred[f] - cuminf_cred_int_overall.lb_pred_10[f]) / N_msm
# 100 * (cuminf_cred_int_overall.mean_pred[f] + cuminf_cred_int_overall.ub_pred_10[f]) / N_msm

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

# exposure[:, "Groups"] = xtickstr
# exposure[:, "Proportion of population"] = pop_sizes ./ N_uk
# exposure[:, "Proportion infected (post. mean)"] = prop_inf
# exposure[:, "Proportion infected (post. 10%)"] = [
#     (cuminf_cred_int.mean_pred[f, 11] - cuminf_cred_int.lb_pred_10[f, 11]) / (N_uk - N_msm)
#     (cuminf_cred_int.mean_pred[f, 1:10] .- cuminf_cred_int.lb_pred_10[f, 1:10]) ./
#     N_msm_grp[:]
# ]
# exposure[:, "Proportion infected (post. 90%)"] = [
#     (cuminf_cred_int.mean_pred[f, 11] + cuminf_cred_int.ub_pred_10[f, 11]) / (N_uk - N_msm)
#     (cuminf_cred_int.mean_pred[f, 1:10] .+ cuminf_cred_int.ub_pred_10[f, 1:10]) ./
#     N_msm_grp[:]
# ]

display(plt_prop)
# CSV.write("projections/population_exposure_" * string(date_exposure) * ".csv", exposure)


## Main figure 1

layout = @layout [a b; c d]
fig1 = plot(
    plt_R_gbmsm,
    plt_R_ngbmsm,
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
