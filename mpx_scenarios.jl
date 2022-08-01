using Distributions, StatsBase, StatsPlots
using LinearAlgebra, RecursiveArrayTools
using OrdinaryDiffEq, ApproxBayes
using JLD2, MLUtils

## Grab UK data and model set up

include("mpxv_datawrangling.jl");
include("setup_model.jl");

##Load posterior draws
# filenames = ["draws1_nrw.jld2", "draws2_nrw.jld2"]
# smcs = [load(filename)["smc_cng_pnt"] for filename in filenames];
# particles = smcs[1].particles
# particles = [particles; smcs[2].particles]
# param_draws = [part.params for part in particles]
smc = load("smc_posterior_draws_vs3_nrw.jld2")["smc_cng_pnt2"]
param_draws = [part.params for part in smc.particles]
params_no_red = map(θ -> [θ[1:(end-2)]; 0.0; 0.0], param_draws)

##Generate predictions with no change

long_wks = [wks; [wks[end] + Day(7 * k) for k = 1:12]]
long_mpxv_wkly = [mpxv_wkly; zeros(12, 2)]

preds_rwc = map(θ -> mpx_sim_function_chp(θ, constants, long_mpxv_wkly)[2], param_draws)
# preds_nored = map(θ -> mpx_sim_function_chp(θ, constants, long_mpxv_wkly)[2], params_no_red)
## Public health emergency effect forecasts

wkly_vaccinations = [zeros(11); 1000; 2000; fill(5000, 24)] * 1.5
plt_vacs = scatter([wks[1] + Day(7 * (k-1)) for k = 1:size(wkly_vaccinations,1)],wkly_vaccinations,
        title="Projected numbers of MPX vaccine doses given",
        lab="",
        ylabel="Number doses per week",
        size=(800, 600), left_margin=5mm,
        guidefont = 14,tickfont = 11,titlefont = 18)

##
chp_t2 = (Date(2022, 7, 23) - Date(2021, 12, 31)).value #Announcement of Public health emergency
inf_duration_red = 0.0
interventions_ensemble = [(trans_red2=rand(Beta(32 / 20, 68 / 20)),#Based on posterior for first change point with extra dispersion
        vac_effectiveness=rand(Uniform(0.8, 0.9)),
        trans_red_other2=rand(Beta(80 / 20, 20 / 20)),
        wkly_vaccinations, chp_t2, inf_duration_red) for i = 1:length(param_draws)]

no_interventions_ensemble = [(trans_red2=0.0,#Based on posterior for first change point with extra dispersion
        vac_effectiveness=0.0,
        trans_red_other2=0.0,
        wkly_vaccinations=zeros(size(long_mpxv_wkly)), chp_t2, inf_duration_red) for i = 1:length(param_draws)]


preds_and_incidence_interventions = map((θ, intervention) -> mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention)[2:3], param_draws, interventions_ensemble)
preds_and_incidence_no_interventions = map((θ, intervention) -> mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention)[2:3], param_draws, no_interventions_ensemble)

##Gather data
d1, d2 = size(mpxv_wkly)

preds = [x[1] for x in preds_and_incidence_interventions]
incidences = [x[2] for x in preds_and_incidence_interventions]
cum_incidences = [cumsum(x[2], dims=1) for x in preds_and_incidence_interventions]
cum_cases = [cumsum(x[1], dims=1) for x in preds_and_incidence_interventions]
cum_cases_forwards = [cumsum(x[1][(d1+1):end, :], dims=1) for x in preds_and_incidence_interventions]

cum_cases_nointervention = [cumsum(x[1], dims=1) for x in preds_and_incidence_no_interventions]
cum_cases_nointervention_forwards = [cumsum(x[1][(d1+1):end, :], dims=1) for x in preds_and_incidence_no_interventions]
cum_incidences_nointervention = [cumsum(x[2], dims=1) for x in preds_and_incidence_no_interventions]

##Simulation projections

cred_int = cred_intervals(preds)
cred_int_rwc = cred_intervals(preds_rwc)

plt = plot(; ylabel="Weekly cases",
        title="UK Monkeypox Case Projections", yscale=:log10,
        legend=:topleft,
        yticks=([1, 2, 11, 101, 1001], [0, 1, 10, 100, 1000]),
        ylims=(0.8, 3001),
        left_margin=5mm,
        size=(800, 600), dpi=250,
        tickfont=11, titlefont=18, guidefont=18, legendfont=11)
plot!(plt, long_wks, cred_int.median_pred .+ 1, ribbon=(cred_int.lb_pred_025, cred_int.ub_pred_025), lw=0,
        color=[1 2], fillalpha=0.3, lab=["Projection: MSM" "Projection: non-MSM"])
plot!(plt, long_wks, cred_int.median_pred .+ 1, ribbon=(cred_int.lb_pred_25, cred_int.ub_pred_25), lw=3,
        color=[1 2], fillalpha=0.3, lab="")
plot!(plt, long_wks[11:end], cred_int_rwc.median_pred[11:end, :], lw=3, ls=:dash,
        color=[1 2], fillalpha=0.3, lab="")

scatter!(plt, wks[(end-1):end], mpxv_wkly[(end-1):end, :] .+ 1,
        lab="",
        ms=6, color=[1 2], shape=:square)
scatter!(plt, wks[1:(end-2)], mpxv_wkly[1:(end-2), :] .+ 1,
        lab=["Data: MSM" "Data: non-MSM"],
        ms=6, color=[1 2])

display(plt)
savefig(plt, "plots/case_projections.png")

##cumulative Incidence plots
# cred_int_cum_incidence = cred_intervals(cum_incidences)
# cred_int_cum_incidence_no_intervention = cred_intervals(cum_incidences_nointervention)
cred_int_cum_incidence = cred_intervals(cum_cases_forwards)
cred_int_cum_incidence_no_intervention = cred_intervals(cum_cases_nointervention_forwards)


total_cases = sum(mpxv_wkly, dims=1)
plt = plot(; ylabel="Cumulative cases",
        title="UK Monkeypox cumulative case projections",#yscale=:log10,
        legend=:topleft,
        yticks=(0:2500:12500, 0:2500:12500),
        left_margin=5mm,
        size=(800, 600), dpi=250,
        tickfont=11, titlefont=18, guidefont=18, legendfont=11)
plot!(plt, long_wks[((d1+1)):end], total_cases .+ cred_int_cum_incidence.median_pred, ribbon=(cred_int_cum_incidence.lb_pred_025, cred_int_cum_incidence.ub_pred_025), lw=0,
        color=[1 2], fillalpha=0.3, lab=["Projection: MSM" "Projection: non-MSM"])

plot!(plt, long_wks[(d1+1):end], total_cases .+ cred_int_cum_incidence.median_pred, ribbon=(cred_int_cum_incidence.lb_pred_25, cred_int_cum_incidence.ub_pred_25), lw=3,
        color=[1 2], fillalpha=0.3, lab="")
plot!(plt, long_wks[(d1+1):end], total_cases .+ cred_int_cum_incidence_no_intervention.median_pred, lw=3, ls=:dash,
        color=[1 2], fillalpha=0.3, lab="")
scatter!(plt, wks, cumsum(mpxv_wkly, dims=1),
        lab=["Data: MSM" "Data: non-MSM"],
        ms=6, color=[1 2])

display(plt)
savefig(plt, "plots/cumcaseprojections.png")
                