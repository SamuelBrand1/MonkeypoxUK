## Idea is to have both fitness and SBM effects in sexual contact

using Distributions, StatsBase, StatsPlots
using LinearAlgebra, RecursiveArrayTools
using OrdinaryDiffEq, ApproxBayes
using JLD2, MCMCChains, Roots, Dates
using CSV, DataFrames, StatsPlots, Plots.PlotMeasures
import MonkeypoxUK

## MSM data with data inference

past_mpxv_data_inferred = CSV.File("data/weekly_data_imputation_2022-09-30.csv",
                                missingstring = "NA") |> DataFrame

colname = "seqn_fit3"
inferred_prop_na_msm = past_mpxv_data_inferred[:, colname] |> x -> x[.~ismissing.(x)]
mpxv_wkly =
    past_mpxv_data_inferred[1:size(inferred_prop_na_msm, 1), ["gbmsm", "nongbmsm"]] .+
    past_mpxv_data_inferred[1:size(inferred_prop_na_msm, 1), "na_gbmsm"] .*
    hcat(inferred_prop_na_msm, 1.0 .- inferred_prop_na_msm) |> Matrix

wks = Date.(past_mpxv_data_inferred.week[1:size(mpxv_wkly, 1)], DateFormat("dd/mm/yyyy"))
                                
# Leave out first two weeks because reporting changed in early May
# mpxv_wkly = mpxv_wkly[3:(end-4), :]
# wks = wks[3:(end-4)]

mpxv_wkly = mpxv_wkly[3:end, :]
wks = wks[3:end]
## Set up model

include("setup_model.jl");

## Define different priors for different models

# Main model
prior_vect_no_ngbmsm_chg = [
    Gamma(1,1), # α_choose 1
    Beta(5, 5), #p_detect  2
    Beta(1, 1), #p_trans  3
    LogNormal(log(0.25), 1), #R0_other 4
    Gamma(3, 1000 / 3),#  M 5
    LogNormal(log(5), 1),#init_scale 6
    Uniform(135, 199),# chp_t 7
    Beta(1.5,1.5),#trans_red 8
    Uniform(0.0,1e-10),#trans_red_other 9
    Beta(1.5,1.5),#trans_red WHO  10 
    Uniform(0.0,1e-10),#trans_red_other WHO 11
]

# Model with only one metapopulation
prior_vect_one_metapop = [
    Uniform(1e-11,1e-10), # α_choose 1
    Beta(5, 5), #p_detect  2
    Beta(1, 1), #p_trans  3
    LogNormal(log(0.25), 1), #R0_other 4
    Gamma(3, 1000 / 3),#  M 5
    LogNormal(log(5), 1),#init_scale 6
    Uniform(135, 199),# chp_t 7
    Beta(1.5,1.5),#trans_red 8
    Uniform(0.0,1e-10),#trans_red_other 9
    Beta(1.5,1.5),#trans_red WHO  10 
    Uniform(0.0,1e-10),#trans_red_other WHO 11
]

# Model with behaviour change for GBMSM and non-GBMSM

prior_vect = [
    Gamma(1,1), # α_choose 1
    Beta(5, 5), #p_detect  2
    Beta(1, 1), #p_trans  3
    LogNormal(log(0.25), 1), #R0_other 4
    Gamma(3, 1000 / 3),#  M 5
    LogNormal(log(5), 1),#init_scale 6
    Uniform(135, 199),# chp_t 7
    Beta(1.5,1.5),#trans_red 8
    Beta(1.5,1.5),#trans_red_other 9
    Beta(1.5,1.5),#trans_red WHO  10 
    Beta(1.5,1.5),#trans_red_other WHO 11
]

# Model with no behaviour change for GBMSM and non-GBMSM

prior_vect_no_bv_cng = [
    Gamma(1,1), # α_choose 1
    Beta(5, 5), #p_detect  2
    Beta(1, 1), #p_trans  3
    LogNormal(log(0.25), 1), #R0_other 4
    Gamma(3, 1000 / 3),#  M 5
    LogNormal(log(5), 1),#init_scale 6
    Uniform(135, 199),# chp_t 7
    Uniform(0.0,1e-10),#trans_red 8
    Uniform(0.0,1e-10),#trans_red_other 9
    Uniform(0.0,1e-10),#trans_red WHO  10 
    Uniform(0.0,1e-10),#trans_red_other WHO 11
]

model_str_to_prior = Dict("no_ngbmsm_chg" => prior_vect_no_ngbmsm_chg,
                            "no_bv_cng" => prior_vect_no_bv_cng,
                            "one_metapop" => prior_vect_one_metapop,
                            "" => prior_vect)

## Choose model

# description_str = "no_ngbmsm_chg" #<---- This is the main model
# description_str = "no_bv_cng" #<---- This is the version of the model with no behavioural change
# description_str = "one_metapop" #<--- This is the version of the model with no metapopulation structure
description_str = "" #<--- this is the older version main model

prior_vect_cng_pnt = model_str_to_prior[description_str]


#Use SBC for defining the ABC error target and generate prior predictive plots

ϵ_target, plt_prc, hist_err = MonkeypoxUK.simulation_based_calibration(
    prior_vect_cng_pnt,
    wks,
    mpxv_wkly,
    constants;
    target_perc = 0.25,
)

setup_cng_pnt = ABCSMC(
    MonkeypoxUK.mpx_sim_function_chp, #simulation function
    length(prior_vect_cng_pnt), # number of parameters
    ϵ_target, #target ϵ derived from simulation based calibration
    Prior(prior_vect_cng_pnt); #Prior for each of the parameters
    ϵ1 = 1000,
    convergence = 0.05,
    nparticles = 2000,
    α = 0.3,
    kernel = gaussiankernel,
    constants = constants,
    maxiterations = 5 * 10^5,
)

##Run ABC and save results   

smc_cng_pnt = runabc(setup_cng_pnt, mpxv_wkly, verbose = true, progress = true)

param_draws = [particle.params for particle in smc_cng_pnt.particles]
@save("posteriors/posterior_param_draws_" * string(wks[end]) * description_str * ".jld2", param_draws)
detected_cases = [particle.other.detected_cases for particle in smc_cng_pnt.particles]
@save("posteriors/posterior_detected_cases_" * string(wks[end]) * description_str * ".jld2", detected_cases)
onsets = [particle.other.onsets for particle in smc_cng_pnt.particles]
@save("posteriors/posterior_onsets_" * string(wks[end]) * description_str * ".jld2", onsets)
incidences = [particle.other.incidence for particle in smc_cng_pnt.particles]
@save("posteriors/posterior_incidences_" * string(wks[end]) * description_str * ".jld2", incidences)
susceptibilities = [particle.other.susceptibility for particle in smc_cng_pnt.particles]
@save("posteriors/posterior_susceptibilities_" * string(wks[end]) * description_str * ".jld2", susceptibilities)
end_states = [particle.other.end_state for particle in smc_cng_pnt.particles]
@save("posteriors/posterior_end_states_" * string(wks[end]) * description_str * ".jld2", end_states)
start_states = [particle.other.start_state for particle in smc_cng_pnt.particles]
@save("posteriors/posterior_start_states_" * string(wks[end]) * description_str * ".jld2", start_states)
begin_vac_states = [particle.other.state_pre_vaccine for particle in smc_cng_pnt.particles]
@save("posteriors/posterior_begin_vac_states_" * string(wks[end]) * description_str * ".jld2", begin_vac_states)
begin_sept_states = [particle.other.state_sept for particle in smc_cng_pnt.particles]
@save("posteriors/posterior_begin_sept_states_" * string(wks[end]) * description_str * ".jld2", begin_sept_states)
vac_effectivenesses = [particle.other.vac_effectiveness for particle in smc_cng_pnt.particles]
@save("posteriors/posterior_vac_effectivenesses_" * string(wks[end]) * description_str * ".jld2", vac_effectivenesses)

##posterior predictive checking - simple plot to see coherence of model with data


post_preds = [part.other.detected_cases for part in smc_cng_pnt.particles]
plt = plot(; ylabel = "Weekly cases", title = "Posterior predictive checking")
for pred in post_preds

    plot!(plt, wks[1:end], pred[1:end,2], lab = "", color = [1 2], alpha = 0.1)
end
scatter!(plt, wks[1:end], mpxv_wkly[1:end,2], lab = ["Data: (MSM)" "Data: (non-MSM)"])#, ylims = (0, 800))
display(plt)
