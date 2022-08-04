using Distributions, StatsBase, StatsPlots
using LinearAlgebra, RecursiveArrayTools
using OrdinaryDiffEq, ApproxBayes
using JLD2

## Grab UK data and model set up

include("mpxv_datawrangling.jl");
include("setup_model.jl");

##Load posterior draws
smc = load("posteriors/smc_posterior_draws_2022-07-25.jld2")["smc_cng_pnt"]
param_draws = [part.params for part in smc.particles]

## Public health emergency effect forecasts
long_wks = [wks; [wks[end] + Day(7 * k) for k = 1:12]]
long_mpxv_wkly = [mpxv_wkly; zeros(12, 2)]
wkly_vaccinations = [zeros(11); 1000; 2000; fill(5000, 12)] * 1.5
plt_vacs = plot([wks[1] + Day(7 * (k - 1)) for k = 1:size(wkly_vaccinations, 1)], wkly_vaccinations,
        title="Projected weekly number of MPX vaccines doses",
        lab="", color=:black, lw=3, yticks=0:1000:8000,
        ylabel="Number doses per week",
        size=(800, 600), left_margin=5mm,
        guidefont=16, tickfont=11, titlefont=18)
display(plt_vacs)
savefig(plt_vacs, "plots/vaccine_rollout.png")

## Fit future change in risk based on  posterior for first change point with extra dispersion
function mom_fit_beta(X, shrinkage,bias_factor)
        x̄ = mean(X)
        v̄ = var(X)
        if v̄ < x̄ * (1 - x̄)
                α = ((x̄^2 * (1 - x̄) / v̄) - x̄) / shrinkage
                β = (((x̄ * (1 - x̄)^2) / v̄) - (1 - x̄)) / shrinkage
                α̂ = bias_factor*α
                β̂ = (1-bias_factor)*α + β
                return Beta(α̂, β̂)
        else
                println("ERROR")
                return nothing
        end
end
trans_red2_prior = mom_fit_beta([θ[9] for θ in param_draws], 1,0.5)
trans_red_other2_prior = mom_fit_beta([θ[10] for θ in param_draws], 1,0.5)

##
chp_t2 = (Date(2022, 7, 23) - Date(2021, 12, 31)).value #Announcement of Public health emergency
inf_duration_red = 0.0

interventions_ensemble = [(trans_red2=rand(trans_red2_prior),
        vac_effectiveness=rand(Uniform(0.7, 0.85)),
        trans_red_other2=rand(trans_red_other2_prior),
        wkly_vaccinations, chp_t2, inf_duration_red) for i = 1:length(param_draws)]

no_interventions_ensemble = [(trans_red2=0.0,
        vac_effectiveness=0.0,
        trans_red_other2=0.0,
        wkly_vaccinations=zeros(size(long_mpxv_wkly)), chp_t2, inf_duration_red) for i = 1:length(param_draws)]

no_red_ensemble = [(trans_red2=0.0,
        vac_effectiveness=rand(Uniform(0.7, 0.85)),
        trans_red_other2=0.0,
        wkly_vaccinations, chp_t2, inf_duration_red) for i = 1:length(param_draws)]

no_vac_ensemble = [(trans_red2=rand(trans_red2_prior),#Based on posterior for first change point with extra dispersion
        vac_effectiveness=rand(Uniform(0.7, 0.85)),
        trans_red_other2=rand(trans_red_other2_prior),
        wkly_vaccinations=zeros(size(long_mpxv_wkly)), chp_t2, inf_duration_red) for i = 1:length(param_draws)]


preds_and_incidence_interventions = map((θ, intervention) -> mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention)[2:4], param_draws, interventions_ensemble)
preds_and_incidence_no_interventions = map((θ, intervention) -> mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention)[2:4], param_draws, no_interventions_ensemble)
preds_and_incidence_no_vaccines = map((θ, intervention) -> mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention)[2:4], param_draws, no_vac_ensemble)
preds_and_incidence_no_redtrans = map((θ, intervention) -> mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention)[2:4], param_draws, no_red_ensemble)

##Gather data
d1, d2 = size(mpxv_wkly)

preds = [x[1] for x in preds_and_incidence_interventions]
preds_nointervention = [x[1] for x in preds_and_incidence_no_interventions]
preds_novacs = [x[1] for x in preds_and_incidence_no_vaccines]
preds_noredtrans = [x[1] for x in preds_and_incidence_no_redtrans]

cum_cases_forwards = [cumsum(x[1][(d1+1):end, :], dims=1) for x in preds_and_incidence_interventions]
cum_cases_nointervention_forwards = [cumsum(x[1][(d1+1):end, :], dims=1) for x in preds_and_incidence_no_interventions]
cum_cases_novaccines_forwards = [cumsum(x[1][(d1+1):end, :], dims=1) for x in preds_and_incidence_no_vaccines]
cum_cases_noredtrans_forwards = [cumsum(x[1][(d1+1):end, :], dims=1) for x in preds_and_incidence_no_redtrans]

##Simulation projections

cred_int = cred_intervals(preds)
cred_int_rwc = cred_intervals(preds_nointervention)
cred_int_nv = cred_intervals(preds_novacs)
cred_int_nr = cred_intervals(preds_noredtrans)

## MSM projections
plt_msm = plot(; ylabel="Weekly cases",
        title="UK Monkeypox Case Projections (MSM)",# yscale=:log10,
        legend=:topleft,
        # yticks=([1, 2, 11, 101, 1001], [0, 1, 10, 100, 1000]),
        # ylims=(0.8, 3001),
        left_margin=5mm,
        size=(800, 600), dpi=250,
        tickfont=11, titlefont=18, guidefont=18, legendfont=11)
plot!(plt_msm, long_wks, cred_int.median_pred[:, 1],
        ribbon=(cred_int.lb_pred_25[:, 1], cred_int.ub_pred_25[:, 1]),
        lw=3,
        color=:black,
        fillalpha=0.2,
        lab="Forecast")
plot!(plt_msm, long_wks[11:end], cred_int_nr.median_pred[11:end, 1],
        ribbon=(cred_int_nr.lb_pred_25[11:end, 1], cred_int_nr.ub_pred_25[11:end, 1]),
        lw=3,
        ls=:dash,
        color=1, fillalpha=0.2, lab="No further behavioural response")
plot!(plt_msm, long_wks[11:end], cred_int_nv.median_pred[11:end, 1],
        ribbon=(cred_int_nv.lb_pred_25[11:end, 1], cred_int_nv.ub_pred_25[11:end, 1]),
        lw=3,
        ls=:dash,
        color=4, fillalpha=0.2, lab="No vaccinations")
plot!(plt_msm, long_wks[11:end], cred_int_rwc.median_pred[11:end, 1],
        ribbon=(cred_int_rwc.lb_pred_25[11:end, 1], cred_int_rwc.ub_pred_25[11:end, 1]),
        lw=3,
        ls=:dash,
        color=2, fillalpha=0.2, lab="Reasonable worst case scenario")
scatter!(plt_msm, wks[(end):end], mpxv_wkly[(end):end, 1],
        lab="",
        ms=6, color=:black, shape=:square)
scatter!(plt_msm, wks[1:(end-1)], mpxv_wkly[1:(end-1), 1],
        lab="Data",
        ms=6, color=:black)

##
plt_nmsm = plot(; ylabel="Weekly cases",
        title="UK Monkeypox Case Projections (non-MSM)",# yscale=:log10,
        legend=:topleft,
        # yticks=([1, 2, 11, 101, 1001], [0, 1, 10, 100, 1000]),
        # ylims=(0.8, 3001),
        left_margin=5mm,
        size=(800, 600), dpi=250,
        tickfont=11, titlefont=18, guidefont=18, legendfont=11)
plot!(plt_nmsm, long_wks, cred_int.median_pred[:, 2],
        ribbon=(cred_int.lb_pred_25[:, 2], cred_int.ub_pred_25[:, 2]),
        lw=3,
        color=:black,
        fillalpha=0.2,
        lab="Forecast")
plot!(plt_nmsm, long_wks[11:end], cred_int_nr.median_pred[11:end, 2],
        ribbon=(cred_int_nr.lb_pred_25[11:end, 2], cred_int_nr.ub_pred_25[11:end, 2]),
        lw=3,
        ls=:dash,
        color=1, fillalpha=0.2, lab="No further behavioural response")
plot!(plt_nmsm, long_wks[11:end], cred_int_nv.median_pred[11:end, 2],
        ribbon=(cred_int_nv.lb_pred_25[11:end, 2], cred_int_nv.ub_pred_25[11:end, 2]),
        lw=3,
        ls=:dash,
        color=4, fillalpha=0.2, lab="No vaccinations")
plot!(plt_nmsm, long_wks[11:end], cred_int_rwc.median_pred[11:end, 2],
        ribbon=(cred_int_rwc.lb_pred_25[11:end, 2], cred_int_rwc.ub_pred_25[11:end, 2]),
        lw=3,
        ls=:dash,
        color=2, fillalpha=0.2, lab="Reasonable worst case scenario")
scatter!(plt_nmsm, wks[(end):end], mpxv_wkly[(end):end, 2],
        lab="",
        ms=6, color=:black, shape=:square)
scatter!(plt_nmsm, wks[1:(end-1)], mpxv_wkly[1:(end-1), 2],
        lab="Data",
        ms=6, color=:black)



##cumulative Incidence plots
# cred_int_cum_incidence = cred_intervals(cum_incidences)
# cred_int_cum_incidence_no_intervention = cred_intervals(cum_incidences_nointervention)
cred_int_cum_incidence = cred_intervals(cum_cases_forwards)
cred_int_cum_incidence_no_intervention = cred_intervals(cum_cases_nointervention_forwards)
cred_int_cum_no_vaccines = cred_intervals(cum_cases_novaccines_forwards)
cred_int_cum_noredtrans = cred_intervals(cum_cases_noredtrans_forwards)


total_cases = sum(mpxv_wkly, dims=1)
plt_cm_msm = plot(; ylabel="Cumulative cases",
        title="UK Monkeypox cumulative case projections (MSM)",#yscale=:log10,
        legend=:topleft,
        yticks=(0:2500:12500, 0:2500:12500),
        left_margin=5mm,
        size=(800, 600), dpi=250,
        tickfont=11, titlefont=18, guidefont=18, legendfont=11)
plot!(plt_cm_msm, long_wks[((d1+1)):end], total_cases[:, 1] .+ cred_int_cum_incidence.median_pred[:, 1],
        ribbon=(cred_int_cum_incidence.lb_pred_25[:, 1], cred_int_cum_incidence.ub_pred_25[:, 1]),
        lw=3,
        color=:black, fillalpha=0.2, lab="Forecast")

plot!(plt_cm_msm, long_wks[((d1+1)):end], total_cases[:, 1] .+ cred_int_cum_noredtrans.median_pred[:, 1],
        ribbon=(cred_int_cum_noredtrans.lb_pred_25[:, 1], cred_int_cum_noredtrans.ub_pred_25[:, 1]),
        lw=3, ls=:dash,
        color=1, fillalpha=0.2, lab="No further behavioural response")
plot!(plt_cm_msm, long_wks[((d1+1)):end], total_cases[:, 1] .+ cred_int_cum_no_vaccines.median_pred[:, 1],
        ribbon=(cred_int_cum_no_vaccines.lb_pred_25[:, 1], cred_int_cum_no_vaccines.ub_pred_25[:, 1]),
        lw=3, ls=:dash,
        color=4, fillalpha=0.2, lab="No vaccinations")

plot!(plt_cm_msm, long_wks[((d1+1)):end], total_cases[:, 1] .+ cred_int_cum_incidence_no_intervention.median_pred[:, 1],
        ribbon=(cred_int_cum_incidence_no_intervention.lb_pred_25[:, 1], cred_int_cum_incidence_no_intervention.ub_pred_25[:, 1]),
        lw=3, ls=:dash,
        color=2, fillalpha=0.2, lab="Reasonable worst case scenario")

scatter!(plt_cm_msm, wks, cumsum(mpxv_wkly[:, 1], dims=1),
        lab="Data",
        ms=6, color=:black)

##
total_cases = sum(mpxv_wkly, dims=1)
plt_cm_nmsm = plot(; ylabel="Cumulative cases",
        title="UK Monkeypox cumulative case projections (non-MSM)",#yscale=:log10,
        legend=:topleft,
        # yticks=(0:2500:12500, 0:2500:12500),
        left_margin=5mm,
        size=(800, 600), dpi=250,
        tickfont=11, titlefont=17, guidefont=18, legendfont=11)
plot!(plt_cm_nmsm, long_wks[((d1+1)):end], total_cases[:, 2] .+ cred_int_cum_incidence.median_pred[:, 2],
        ribbon=(cred_int_cum_incidence.lb_pred_25[:, 2], cred_int_cum_incidence.ub_pred_25[:, 2]),
        lw=3,
        color=:black, fillalpha=0.2, lab="Forecast")

plot!(plt_cm_nmsm, long_wks[((d1+1)):end], total_cases[:, 2] .+ cred_int_cum_noredtrans.median_pred[:, 2],
        ribbon=(cred_int_cum_noredtrans.lb_pred_25[:, 2], cred_int_cum_noredtrans.ub_pred_25[:, 2]),
        lw=3, ls=:dash,
        color=1, fillalpha=0.2, lab="No further behavioural response")
plot!(plt_cm_nmsm, long_wks[((d1+1)):end], total_cases[:, 2] .+ cred_int_cum_no_vaccines.median_pred[:, 2],
        ribbon=(cred_int_cum_no_vaccines.lb_pred_25[:, 2], cred_int_cum_no_vaccines.ub_pred_25[:, 2]),
        lw=3, ls=:dash,
        color=4, fillalpha=0.2, lab="No vaccinations")

plot!(plt_cm_nmsm, long_wks[((d1+1)):end], total_cases[:, 2] .+ cred_int_cum_incidence_no_intervention.median_pred[:, 2],
        ribbon=(cred_int_cum_incidence_no_intervention.lb_pred_25[:, 2], cred_int_cum_incidence_no_intervention.ub_pred_25[:, 2]),
        lw=3, ls=:dash,
        color=2, fillalpha=0.2, lab="Reasonable worst case scenario")

scatter!(plt_cm_nmsm, wks, cumsum(mpxv_wkly[:, 2], dims=1),
        lab="Data",
        ms=6, color=:black)

##Combined plot for cases
lo = @layout [a b; c d]
plt = plot(plt_msm, plt_nmsm, plt_cm_msm, plt_cm_nmsm,
        size=(1600, 1200), dpi=250,
        left_margin=10mm,
        bottom_margin=10mm,
        right_margin=10mm,
        layout=lo)
display(plt)
savefig(plt, "plots/case_projections_" * string(wks[end]) * ".png")

## Change in transmission over time

prob_trans = [θ[4] for θ in param_draws]
red_sx_trans = [θ[9] for θ in param_draws]
chp1 = [θ[8] for θ in param_draws]
red_sx_trans2 = [int.trans_red2 for int in interventions_ensemble]

function generate_trans_risk_over_time(p_trans, trans_red, trans_red2, chp, chp2, ts)
        log_p = [log(p_trans) + log(1 - trans_red) * (t >= chp) + log(1 - trans_red2) * (t >= chp2) for t in ts]
        return exp.(log_p)
end
ts_risk = 1:365
p_sx_trans_risks = map((p_tr, red_sx_tr, red_sx_tr2, ch1) -> generate_trans_risk_over_time(p_tr, red_sx_tr, red_sx_tr2, ch1, chp_t2, ts_risk),
        prob_trans, red_sx_trans, red_sx_trans2, chp1) .|> x -> reshape(x, length(x), 1)
p_sx_trans_risks_rwc = map((p_tr, red_sx_tr, red_sx_tr2, ch1) -> generate_trans_risk_over_time(p_tr, red_sx_tr, red_sx_tr2, ch1, Inf, ts_risk),
        prob_trans, red_sx_trans, red_sx_trans2, chp1) .|> x -> reshape(x, length(x), 1)

sx_trans_risk_cred_int = prev_cred_intervals(p_sx_trans_risks)
sx_trans_risk_cred_no_int = prev_cred_intervals(p_sx_trans_risks_rwc)

dates = [Date(2021, 12, 31) + Day(t) for t in ts_risk]
f = findfirst(dates .== Date(2022, 7, 23))

plt_chng = plot(dates, sx_trans_risk_cred_int.median_pred,
        ribbon=(sx_trans_risk_cred_int.lb_pred_25, sx_trans_risk_cred_int.ub_pred_25),
        lw=3, fillalpha=0.2,
        lab="Transmission probability (forecast)",
        title="Transmission probability (sexual contacts)",
        ylabel="Prob. per sexual contact",
        xlims=(long_wks[1] - Day(7), long_wks[end] - Day(7)),
        left_margin=5mm,
        size=(800, 600), dpi=250,
        tickfont=11, titlefont=17, guidefont=18, legendfont=11)

plot!(plt_chng, dates[f:end], sx_trans_risk_cred_no_int.median_pred[f:end],
        ribbon=(sx_trans_risk_cred_no_int.lb_pred_25[f:end], sx_trans_risk_cred_no_int.ub_pred_25[f:end]),
        lw=3, fillalpha=0.3,
        lab="Transmission probability (no change)")
vline!(plt_chng, [Date(2022, 7, 23)],
        lw=4, color=:black, ls=:dash,
        lab="WHO declaration")
##
R0_other = [θ[5] for θ in param_draws]
red_oth_trans = [θ[10] for θ in param_draws]
chp1 = [θ[8] for θ in param_draws]
red_oth_trans2 = [int.trans_red_other2 for int in interventions_ensemble]

p_oth_trans_risks = map((p_tr, red_sx_tr, red_sx_tr2, ch1) -> generate_trans_risk_over_time(p_tr, red_sx_tr, red_sx_tr2, ch1, chp_t2, ts_risk),
        R0_other, red_oth_trans, red_oth_trans2, chp1) .|> x -> reshape(x, length(x), 1)
p_oth_trans_risks_rwc = map((p_tr, red_sx_tr, red_sx_tr2, ch1) -> generate_trans_risk_over_time(p_tr, red_sx_tr, red_sx_tr2, ch1, Inf, ts_risk),
        R0_other, red_oth_trans, red_oth_trans2, chp1) .|> x -> reshape(x, length(x), 1)

oth_sx_trans_risk_cred_int = prev_cred_intervals(p_oth_trans_risks)
oth_trans_risk_cred_no_int = prev_cred_intervals(p_oth_trans_risks_rwc)

plt_chng_oth = plot(dates, oth_sx_trans_risk_cred_int.median_pred,
        ribbon=(oth_sx_trans_risk_cred_int.lb_pred_25, oth_sx_trans_risk_cred_int.ub_pred_25),
        lw=3, fillalpha=0.2,
        lab="R0, non-sexual (forecast)",
        title="Reproductive number (non-sexual contacts)",
        ylabel="R(t) (non-sexual contacts)",
        xlims=(long_wks[1] - Day(7), long_wks[end] - Day(7)),
        left_margin=5mm,
        size=(800, 600), dpi=250,
        tickfont=11, titlefont=17, guidefont=18, legendfont=11)

plot!(plt_chng_oth, dates[f:end], oth_trans_risk_cred_no_int.median_pred[f:end],
        ribbon=(oth_trans_risk_cred_no_int.lb_pred_25[f:end], oth_trans_risk_cred_no_int.ub_pred_25[f:end]),
        lw=3, fillalpha=0.3,
        lab="R0, non-sexual (no change)")
vline!(plt_chng_oth, [Date(2022, 7, 23)],
        lw=4, color=:black, ls=:dash,
        lab="WHO declaration")
## Combined plot

plt = plot(plt_chng, plt_chng_oth,
        size=(1600, 800), dpi=250,
        left_margin=10mm,
        bottom_margin=10mm,
        right_margin=10mm,
        layout=(1, 2))
display(plt)
savefig(plt, "plots/risk_over_time" * string(wks[end]) * ".png")


##Prevalence plots

prev_cred_int = prev_cred_intervals([pred[3] for pred in preds_and_incidence_interventions])
prev_cred_no_int = prev_cred_intervals([pred[3] for pred in preds_and_incidence_no_interventions])
prev_cred_int_overall = prev_cred_intervals([sum(pred[3][:, 1:10], dims=2) for pred in preds_and_incidence_interventions])
prev_cred_no_int_overall = prev_cred_intervals([sum(pred[3][:, 1:10], dims=2) for pred in preds_and_incidence_no_interventions])


N_msm_grp = N_msm .* ps'
_wks = long_wks .- Day(7)
plt_prev = plot(_wks, prev_cred_int.median_pred[:, 1:10] ./ N_msm_grp,
        ribbon=(prev_cred_int.lb_pred_25[:, 1:10] ./ N_msm_grp, prev_cred_int.ub_pred_25[:, 1:10] ./ N_msm_grp),
        lw=[j / 1.5 for i = 1:1, j = 1:10],
        lab=hcat(["Forecast"], fill("", 1, 9)),
        yticks=(0:0.02:0.18, [string(round(y * 100)) * "%" for y in 0:0.02:0.18]),
        ylabel="Prevalence",
        title="MPX prevalence by sexual activity group (MSM)",
        color=:black,
        fillalpha=0.1,
        left_margin=5mm,
        size=(800, 600), dpi=250,
        tickfont=11, titlefont=18, guidefont=18, legendfont=11)

plot!(plt_prev, _wks[11:end], prev_cred_no_int.median_pred[11:end, 1:10] ./ N_msm_grp,
        lw=[j / 1.5 for i = 1:1, j = 1:10],
        ls=:dash,
        color=:black,
        lab=hcat(["Worst case scenario"], fill("", 1, 9)))
plot!(plt_prev, _wks, prev_cred_int_overall.median_pred ./ N_msm,
        ribbon=(prev_cred_int_overall.lb_pred_25 ./ N_msm, prev_cred_int_overall.ub_pred_25 ./ N_msm),
        lw=3,
        color=1,
        fillalpha=0.1,
        lab="",
        xticks=[Date(2022, 6, 1), Date(2022, 8, 1), Date(2022, 10, 1)],
        yticks=(0:0.001:0.0050, [string(y * 100) * "%" for y in 0:0.001:0.0050]),
        inset=bbox(0.25, 0.125, 0.25, 0.25, :top, :left),
        xtickfont=7,
        subplot=2,
        grid=nothing,
        title="Overall")
plot!(plt_prev, _wks, prev_cred_no_int_overall.median_pred ./ N_msm,
        subplot=2,
        color=1, ls=:dash, lw=3, lab="")
##
plt_prev_overall = plot(_wks, prev_cred_int.median_pred[:, 11] ./ (N_uk - N_msm),
        ribbon=(prev_cred_int.lb_pred_25[:, 11] ./ (N_uk - N_msm), prev_cred_int.ub_pred_25[:, 11] ./ (N_uk - N_msm)),
        lw=3,
        lab="Forecast",
        ylabel="Prevalence",
        title="MPX Prevalence (non-MSM)",
        color=2,
        yticks=(0:5e-7:4.0e-6, [string(round(y * 100, digits=10)) * "%" for y in 0:5e-7:4.0e-6]),
        fillalpha=0.2,
        left_margin=5mm,
        size=(800, 600), dpi=250,
        tickfont=11, titlefont=18, guidefont=18, legendfont=11)
plot!(plt_prev_overall, _wks[11:end], prev_cred_no_int.median_pred[11:end, 11] ./ (N_uk - N_msm),
        color=2, ls=:dash, lw=3, lab="Worst case scenario")
##
plt = plot(plt_chng, plt_chng_oth,plt_prev, plt_prev_overall,
        size=(1750, 1600), dpi=250,
        left_margin=10mm,
        bottom_margin=10mm,
        right_margin=10mm,
        layout=(2, 2))
display(plt)
savefig(plt, "plots/change_and_prevalence" * string(wks[end]) * ".png")

## Mean sexual contacts of a detected person

function sx_contacts_msm(pred, mean_daily_cnts)
        sum(mean_daily_cnts' .* pred[:, 1:10] ./ (sum(pred[:, 1:10], dims=2) .+ 1e-10), dims=2) #Very slight perturbation to avoid NaN
end

observed_case_sx_cnt_rates = map(pred -> sx_contacts_msm(pred, mean_daily_cnts), [pred[3] for pred in preds_and_incidence_no_interventions])
obs_case_sx_cnt_rates_pred = prev_cred_intervals(observed_case_sx_cnt_rates)

plot(long_wks .- Week(1), obs_case_sx_cnt_rates_pred.median_pred,
        ribbon=(obs_case_sx_cnt_rates_pred.lb_pred_25, obs_case_sx_cnt_rates_pred.ub_pred_25),
        title="Typical sexual contact (cases)", fillalpha=0.2,
        lab="",
        left_margin=5mm,
        size=(800, 600), dpi=250,
        tickfont=11, titlefont=18, guidefont=18, legendfont=11)
