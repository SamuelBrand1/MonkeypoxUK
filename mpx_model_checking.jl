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


param_draws =
        load("posteriors/posterior_param_draws_2022-09-26no_ngbmsm_chg.jld2")["param_draws"]
detected_cases =
    load("posteriors/posterior_detected_cases_2022-09-26no_ngbmsm_chg.jld2")["detected_cases"]


incidences = load("posteriors/posterior_incidences_2022-09-26no_ngbmsm_chg.jld2")["incidences"]

##

resampled_cases = map(incidences, param_draws) do inc, θ
    p_detect = θ[2]
    M = θ[5]
    grouped_incidences = [sum(inc[:,1:10], dims = 2) inc[:,11]]
    
    return map(n -> rand(BetaBinomial(n, p_detect * M, (1-p_detect) * M)), grouped_incidences)
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

marginal_posteriors = zeros(size(above_below[1]))

for i = 1:size(marginal_posteriors,1), j = 1:size(marginal_posteriors,2)
    marginal_posteriors[i,j] = mean([x[i,j] >= 0 for x in above_below])
end

pit_histo = histogram(marginal_posteriors[:],
            norm = :pdf,
            lab = "",
            bins = 20,
            title = "Posterior marginal predictive p-values")
hline!(pit_histo, [1],
        lab = "",
        lw = 3)            
pit_qq = qqplot(Uniform,marginal_posteriors[:],
        title = "Posterior marginal predictive QQ-plot",
        lw = 3)

plt_pit = plot(pit_histo,pit_qq,
                size = (1000,400),
                dpi = 250)        
# savefig(pit_histo, "plots/pit_histo.png")
# savefig(pit_qq, "plots/pit_qq.png")
savefig(plt_pit, "plots/pit_plots.png")