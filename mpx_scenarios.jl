using Distributions, StatsBase, StatsPlots
using Plots.PlotMeasures
using LinearAlgebra, RecursiveArrayTools
using OrdinaryDiffEq, ApproxBayes,MCMCChains
using JLD2
using MonkeypoxUK

## Grab UK data and model set up
include("mpxv_datawrangling_inff.jl");
include("setup_model.jl");

##Load posterior draws
# param_draws1 = load("posteriors/posterior_param_draws_2022-08-26_vs1.jld2")["param_draws"]
# param_draws2 = load("posteriors/posterior_param_draws_2022-08-26_vs2.jld2")["param_draws"]
param_draws = mapreduce(fn -> load(fn)["param_draws"], vcat, ["posteriors/posterior_param_draws_2022-08-26_vs1.jld2","posteriors/posterior_param_draws_2022-08-26_vs2.jld2"])
# param_draws = load("posteriors/posterior_param_draws_2022-08-26_vs1prc.jld2")["param_draws"]
## Public health emergency effect forecasts
long_wks = [wks; [wks[end] + Day(7 * k) for k = 1:12]]
long_mpxv_wkly = [mpxv_wkly; zeros(12, 2)]
wkly_vaccinations_ceased = [copy(wkly_vaccinations)[1:length(wks)+1];fill(0,18)]

plt_vacs = plot([wks[1] + Day(7 * (k - 1)) for k = 1:size(long_wks, 1)], cumsum(wkly_vaccinations)[1:size(long_mpxv_wkly,1)],
        title="Cumulative number of MPX vaccine doses",
        lab="Projection", color=:black, lw=3,# yticks=0:1000:8000,
        ylabel="Cum. doses",
        # xlims = (Date(2022,5,1), Date(2022,12,31)),
        size=(800, 600), left_margin=5mm,
        guidefont=16, tickfont=11, titlefont=18,legendfont = 16,
        legend = :topleft,
        right_margin = 5mm)

plot!(plt_vacs,[wks[1] + Day(7 * (k - 1)) for k = 1:size(long_wks, 1)],cumsum(wkly_vaccinations_ceased)[1:size(long_wks, 1)],
                lab = "Vaccine ceased scenario",
                lw = 3,
                color = :black,
                ls = :dot)        
scatter!(plt_vacs,[Date(2022,8,30)],[38_079],
                lab = "UKHSA reported vaccines",
                ms = 8)        
display(plt_vacs)
savefig(plt_vacs, "plots/vaccine_rollout.png")


##
chp_t2 = (Date(2022, 7, 23) - Date(2021, 12, 31)).value #Announcement of Public health emergency
inf_duration_red = 0.0

interventions_ensemble = [(trans_red2=θ[9] * θ[11],
        vac_effectiveness=rand(Uniform(0.7, 0.85)),
        trans_red_other2=θ[10] * θ[12],
        wkly_vaccinations, chp_t2, inf_duration_red) for θ in param_draws]

# no_interventions_ensemble = [(trans_red2=0.0,
#         vac_effectiveness=0.0,
#         trans_red_other2=0.0,
#         wkly_vaccinations = wkly_vaccinations_ceased, chp_t2, inf_duration_red) for θ in param_draws]

# no_red_ensemble = [(trans_red2=0.0,
#         vac_effectiveness=rand(Uniform(0.7, 0.85)),
#         trans_red_other2=0.0,
#         wkly_vaccinations, chp_t2, inf_duration_red) for θ in param_draws]

no_vac_ensemble = [(trans_red2 =θ[9] * θ[11],#Based on posterior for first change point with extra dispersion
        vac_effectiveness = rand(Uniform(0.7, 0.85)),
        trans_red_other2 = θ[10] * θ[12],
        wkly_vaccinations = wkly_vaccinations_ceased,
        chp_t2, 
        inf_duration_red) for θ in param_draws]

mpx_sim_function_interventions = MonkeypoxUK.mpx_sim_function_interventions

preds_and_incidence_interventions = map((θ, intervention) -> mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention)[2:4], param_draws, interventions_ensemble)
preds_and_incidence_interventions_4wkrev = map((θ, intervention) -> mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention,4)[2:4], param_draws, interventions_ensemble)
preds_and_incidence_interventions_12wkrev = map((θ, intervention) -> mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention,12)[2:4], param_draws, interventions_ensemble)

preds_and_incidence_interventions_cvac = map((θ, intervention) -> mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention)[2:4], param_draws, no_vac_ensemble)
preds_and_incidence_interventions_cvac_4wkrev = map((θ, intervention) -> mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention,4)[2:4], param_draws, no_vac_ensemble)
preds_and_incidence_interventions_cvac_12wkrev = map((θ, intervention) -> mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention,12)[2:4], param_draws, no_vac_ensemble)

# preds_and_incidence_no_interventions = map((θ, intervention) -> mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention)[2:4], param_draws, no_interventions_ensemble)
# preds_and_incidence_no_vaccines = map((θ, intervention) -> mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention)[2:4], param_draws, no_vac_ensemble)
# preds_and_incidence_no_redtrans = map((θ, intervention) -> mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention)[2:4], param_draws, no_red_ensemble)

## Cumulative incidence on week 12 (Mid July) for highest frequency groups for paper
cum_inc_wk_12 = [sum(pred[2][1:15,3:10])./sum(N_msm*ps[3:10]) for pred in preds_and_incidence_interventions]
mean(cum_inc_wk_12)
@show mean(cum_inc_wk_12),quantile(cum_inc_wk_12,[0.1,0.9])

##Gather data
d1, d2 = size(mpxv_wkly)

preds = [x[1] for x in preds_and_incidence_interventions]
preds_4wk = [x[1] for x in preds_and_incidence_interventions_4wkrev]
preds_12wk = [x[1] for x in preds_and_incidence_interventions_12wkrev]

preds_cvac = [x[1] for x in preds_and_incidence_interventions_cvac]
preds_cvac4wk = [x[1] for x in preds_and_incidence_interventions_cvac_4wkrev]
preds_cvac12wk = [x[1] for x in preds_and_incidence_interventions_cvac_12wkrev]

# preds_nointervention = [x[1] for x in preds_and_incidence_no_interventions]
# preds_novacs = [x[1] for x in preds_and_incidence_no_vaccines]
# preds_noredtrans = [x[1] for x in preds_and_incidence_no_redtrans]

cum_cases_forwards = [cumsum(x[1][(d1+1):end, :], dims=1) for x in preds_and_incidence_interventions]
cum_cases_forwards_4wk = [cumsum(x[1][(d1+1):end, :], dims=1) for x in preds_and_incidence_interventions_4wkrev]
cum_cases_forwards_12wk = [cumsum(x[1][(d1+1):end, :], dims=1) for x in preds_and_incidence_interventions_12wkrev]

cum_cases_forwards_cvac = [cumsum(x[1][(d1+1):end, :], dims=1) for x in preds_and_incidence_interventions_cvac]
cum_cases_forwards_cvac4wk = [cumsum(x[1][(d1+1):end, :], dims=1) for x in preds_and_incidence_interventions_cvac_4wkrev]
cum_cases_forwards_cvac12wk = [cumsum(x[1][(d1+1):end, :], dims=1) for x in preds_and_incidence_interventions_cvac_12wkrev]

# cum_cases_nointervention_forwards = [cumsum(x[1][(d1+1):end, :], dims=1) for x in preds_and_incidence_no_interventions]
# cum_cases_novaccines_forwards = [cumsum(x[1][(d1+1):end, :], dims=1) for x in preds_and_incidence_no_vaccines]
# cum_cases_noredtrans_forwards = [cumsum(x[1][(d1+1):end, :], dims=1) for x in preds_and_incidence_no_redtrans]


##Simulation projections
long_wks = [wks; [wks[end] + Day(7 * k) for k = 1:12]]
long_mpxv_wkly = [mpxv_wkly; zeros(12, 2)]

cred_int = MonkeypoxUK.cred_intervals(preds)
cred_int_4wk = MonkeypoxUK.cred_intervals(preds_4wk)
cred_int_12wk = MonkeypoxUK.cred_intervals(preds_12wk)

cred_int_cvac = MonkeypoxUK.cred_intervals(preds_cvac)
cred_int_cvac4wk = MonkeypoxUK.cred_intervals(preds_cvac4wk)
cred_int_cvac12wk = MonkeypoxUK.cred_intervals(preds_cvac12wk)
# cred_int_rwc = MonkeypoxUK.cred_intervals(preds_nointervention)
# cred_int_nv = MonkeypoxUK.cred_intervals(preds_novacs)
# cred_int_nr = MonkeypoxUK.cred_intervals(preds_noredtrans)

## MSM projections
d_proj = 19

plt_msm = plot(; ylabel="Weekly cases",
        title="UK Monkeypox Case Projections (MSM)",# yscale=:log10,
        legend=:topright,
        # yticks=([1, 2, 11, 101, 1001], [0, 1, 10, 100, 1000]),
        # ylims=(-5, 700),
        xticks=([Date(2022, 5, 1) + Month(k) for k = 0:6], [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:6]),
        left_margin=5mm,
        size=(800, 600), dpi=250,
        tickfont=11, titlefont=18, guidefont=18, legendfont=11)

plot!(plt_msm, long_wks, cred_int_12wk.mean_pred[:, 1],
        ribbon=(cred_int_12wk.lb_pred_10[:, 1], cred_int_12wk.ub_pred_10[:, 1]),
        lw=3,
        color=:black,
        fillalpha=0.2,
        lab="12 week reversion")
plot!(plt_msm, long_wks[d_proj:end], cred_int_cvac12wk.mean_pred[d_proj:end, 1],
        # ribbon=(cred_int_cvac.lb_pred_10[d_proj:end, 1], cred_int_cvac.ub_pred_10[d_proj:end, 1]),
        lw=3,
        color=:black,
        ls = :dash,
        fillalpha=0.2,
        lab="12 week reversion (vaccine rollout ceases)")        

plot!(plt_msm, long_wks[19:end], cred_int_4wk.mean_pred[19:end, 1],
        ribbon=(cred_int_4wk.lb_pred_10[19:end, 1], cred_int_4wk.ub_pred_10[19:end, 1]),
        lw=3,
        color=2, fillalpha=0.2, lab="4 week reversion")

plot!(plt_msm, long_wks[19:end], cred_int_cvac4wk.mean_pred[19:end, 1],
        # ribbon=(cred_int_4wk.lb_pred_10[19:end, 1], cred_int_4wk.ub_pred_10[19:end, 1]),
        lw=3,
        ls = :dash,
        color=2, fillalpha=0.2, lab="4 week reversion (vaccine rollout ceases)")        
# plot!(plt_msm, long_wks[11:end], cred_int_nv.mean_pred[11:end, 1],
#         ribbon=(cred_int_nv.lb_pred_10[11:end, 1], cred_int_nv.ub_pred_10[11:end, 1]),
#         lw=3,
#         ls=:dash,
#         color=4, fillalpha=0.2, lab="No vaccinations")
# plot!(plt_msm, long_wks[11:end], cred_int_rwc.mean_pred[11:end, 1],
#         ribbon=(cred_int_rwc.lb_pred_10[11:end, 1], cred_int_rwc.ub_pred_10[11:end, 1]),
#         lw=3,
#         ls=:dash,
#         color=2, fillalpha=0.2, lab="Reasonable worst case scenario")

scatter!(plt_msm, wks[(end):end], mpxv_wkly[(end):end, 1],
        lab="",
        ms=6, color=:black, shape=:square)
scatter!(plt_msm, wks[3:(end-1)], mpxv_wkly[3:(end-1), 1],
        lab="Data",
        ms=6, color=:black)

##
plt_nmsm = plot(; ylabel="Weekly cases",
        title="UK Monkeypox Case Projections (non-MSM)",# yscale=:log10,
        legend=:topright,
        # yticks=([1, 2, 11, 101, 1001], [0, 1, 10, 100, 1000]),
        ylims=(-1, 50),
        xticks=([Date(2022, 5, 1) + Month(k) for k = 0:6], [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:6]),
        left_margin=5mm,
        size=(800, 600), dpi=250,
        tickfont=11, titlefont=18, guidefont=18, legendfont=11)
        
plot!(plt_nmsm, long_wks, cred_int_12wk.mean_pred[:, 2],
        ribbon=(cred_int_12wk.lb_pred_10[:, 2], cred_int_12wk.ub_pred_10[:, 2]),
        lw=3,
        color=:black,
        fillalpha=0.2,
        lab="12 week reversion")

plot!(plt_nmsm, long_wks[d_proj:end], cred_int_cvac12wk.mean_pred[d_proj:end, 2],
        # ribbon=(cred_int_cvac.lb_pred_10[d_proj:end, 1], cred_int_cvac.ub_pred_10[d_proj:end, 1]),
        lw=3,
        color=:black,
        ls = :dash,
        fillalpha=0.2,
        lab="12 week reversion (vaccine rollout ceases)")        

plot!(plt_nmsm, long_wks[19:end], cred_int_4wk.mean_pred[19:end, 2],
        ribbon=(cred_int_4wk.lb_pred_10[19:end, 2], cred_int_4wk.ub_pred_10[19:end, 2]),
        lw=3,
        color=2, fillalpha=0.2, lab="4 week reversion")

plot!(plt_nmsm, long_wks[19:end], cred_int_cvac4wk.mean_pred[19:end, 2],
        # ribbon=(cred_int_4wk.lb_pred_10[19:end, 1], cred_int_4wk.ub_pred_10[19:end, 1]),
        lw=3,
        ls = :dash,
        color=2, fillalpha=0.2, lab="4 week reversion (vaccine rollout ceases)")        
# plot!(plt_msm, long_wks[11:end], cred_int_nv.mean_pred[11:end, 1],
#         ribbon=(cred_int_nv.lb_pred_10[11:end, 1], cred_int_nv.ub_pred_10[11:end, 1]),
#         lw=3,
#         ls=:dash,
#         color=4, fillalpha=0.2, lab="No vaccinations")
# plot!(plt_msm, long_wks[11:end], cred_int_rwc.mean_pred[11:end, 1],
#         ribbon=(cred_int_rwc.lb_pred_10[11:end, 1], cred_int_rwc.ub_pred_10[11:end, 1]),
#         lw=3,
#         ls=:dash,
#         color=2, fillalpha=0.2, lab="Reasonable worst case scenario")

scatter!(plt_nmsm, wks[(end):end], mpxv_wkly[(end):end, 2],
        lab="",
        ms=6, color=:black, shape=:square)
scatter!(plt_nmsm, wks[3:(end-1)], mpxv_wkly[3:(end-1), 2],
        lab="Data",
        ms=6, color=:black)

##cumulative Incidence plots

cred_int_cum_incidence = MonkeypoxUK.cred_intervals(cum_cases_forwards)
cred_int_cum_incidence4wks = MonkeypoxUK.cred_intervals(cum_cases_forwards_4wk)
cred_int_cum_incidence12wks = MonkeypoxUK.cred_intervals(cum_cases_forwards_12wk)

cred_int_cum_incidence_cvac = MonkeypoxUK.cred_intervals(cum_cases_forwards_cvac)
cred_int_cum_incidence_cvac4wks = MonkeypoxUK.cred_intervals(cum_cases_forwards_cvac4wk)
cred_int_cum_incidence_cvac12wks = MonkeypoxUK.cred_intervals(cum_cases_forwards_cvac12wk)



total_cases = sum(mpxv_wkly, dims=1)

plt_cm_msm = plot(; ylabel="Cumulative cases",
        title="UK Monkeypox cumulative case projections (MSM)",#yscale=:log10,
        legend=:topleft,
        # yticks=(0:2500:12500, 0:2500:100),
        xticks=([Date(2022, 5, 1) + Month(k) for k = 0:6], [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:6]),
        left_margin=5mm,
        size=(800, 600), dpi=250,
        tickfont=11, titlefont=18, guidefont=18, legendfont=11)

plot!(plt_cm_msm, long_wks[((d1+1)):end], total_cases[:, 1] .+ cred_int_cum_incidence12wks.mean_pred[:, 1],
        ribbon=(cred_int_cum_incidence12wks.lb_pred_10[:, 1], cred_int_cum_incidence12wks.ub_pred_10[:, 1]),
        lw=3,
        color=:black, fillalpha=0.2, lab="12 week reversion")

plot!(plt_cm_msm, long_wks[((d1+1)):end], total_cases[:, 1] .+ cred_int_cum_incidence_cvac12wks.mean_pred[:, 1],
        ribbon=(cred_int_cum_incidence_cvac12wks.lb_pred_10[:, 1], cred_int_cum_incidence_cvac12wks.ub_pred_10[:, 1]),
        lw=3, ls=:dash,
        color=:black, fillalpha=0.2, lab="12 week reversion (vaccine rollout ceases)")

plot!(plt_cm_msm, long_wks[((d1+1)):end], total_cases[:, 1] .+ cred_int_cum_incidence4wks.mean_pred[:, 1],
        ribbon=(cred_int_cum_incidence4wks.lb_pred_10[:, 1], cred_int_cum_incidence4wks.ub_pred_10[:, 1]),
        lw=3,
        color=2, fillalpha=0.2, lab="4 week reversion")

plot!(plt_cm_msm, long_wks[((d1+1)):end], total_cases[:, 1] .+ cred_int_cum_incidence_cvac4wks.mean_pred[:, 1],
        ribbon=(cred_int_cum_incidence_cvac4wks.lb_pred_10[:, 1], cred_int_cum_incidence_cvac4wks.ub_pred_10[:, 1]),
        lw=3, ls=:dash,
        color=2, fillalpha=0.2, lab="4 week reversion (vaccine rollout ceases)")




scatter!(plt_cm_msm, wks, cumsum(mpxv_wkly[:, 1], dims=1),
        lab="Data",
        ms=6, color=:black)

##
total_cases = sum(mpxv_wkly, dims=1)

plt_cm_nmsm = plot(; ylabel="Cumulative cases",
        title="UK Monkeypox cumulative case projections (non-MSM)",#yscale=:log10,
        legend=:topleft,
        # yticks=(0:2500:12500, 0:2500:12500),
        xticks=([Date(2022, 5, 1) + Month(k) for k = 0:6], [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:6]),
        left_margin=5mm,
        size=(800, 600), dpi=250,
        tickfont=11, titlefont=17, guidefont=18, legendfont=11)

plot!(plt_cm_nmsm, long_wks[((d1+1)):end], total_cases[:, 2] .+ cred_int_cum_incidence12wks.mean_pred[:, 2],
        ribbon=(cred_int_cum_incidence12wks.lb_pred_10[:, 2], cred_int_cum_incidence12wks.ub_pred_10[:, 2]),
        lw=3,
        color=:black, fillalpha=0.2, lab="12 week reversion")

plot!(plt_cm_nmsm, long_wks[((d1+1)):end], total_cases[:, 2] .+ cred_int_cum_incidence_cvac12wks.mean_pred[:, 2],
        ribbon=(cred_int_cum_incidence_cvac12wks.lb_pred_10[:, 2], cred_int_cum_incidence_cvac12wks.ub_pred_10[:, 2]),
        lw=3, ls=:dash,
        color=:black, fillalpha=0.2, lab="12 week reversion (vaccine rollout ceases)")

plot!(plt_cm_nmsm, long_wks[((d1+1)):end], total_cases[:, 2] .+ cred_int_cum_incidence4wks.mean_pred[:, 2],
        ribbon=(cred_int_cum_incidence4wks.lb_pred_10[:, 2], cred_int_cum_incidence4wks.ub_pred_10[:, 2]),
        lw=3,
        color=2, fillalpha=0.2, lab="4 week reversion")

plot!(plt_cm_nmsm, long_wks[((d1+1)):end], total_cases[:, 2] .+ cred_int_cum_incidence_cvac4wks.mean_pred[:, 2],
        ribbon=(cred_int_cum_incidence_cvac4wks.lb_pred_10[:, 2], cred_int_cum_incidence_cvac4wks.ub_pred_10[:, 2]),
        lw=3, ls=:dash,
        color=2, fillalpha=0.2, lab="4 week reversion (vaccine rollout ceases)")



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
# savefig(plt, "plots/case_projections_ukhsa.png")


## Change in transmission over time

prob_trans = [θ[4] for θ in param_draws]
red_sx_trans = [θ[9] for θ in param_draws]
chp1 = [θ[8] for θ in param_draws]
red_sx_trans2 = [int.trans_red2 for int in interventions_ensemble]

function generate_trans_risk_over_time(p_trans, trans_red, trans_red2, chp, chp2, ts)
        log_p = [log(p_trans) + log(1 - trans_red) * (t >= chp) + log(1 - trans_red2) * (t >= chp2) for t in ts]
        return exp.(log_p)
end

function generate_trans_risk_over_time(p_trans, trans_red, trans_red2, chp, chp2, ts, days_reversion;reversion_time = (Date(2022,9,1) - Date(2021,12,31)).value )
        log_r = -(log(1 - trans_red) + log(1 - trans_red2))/days_reversion 
        log_p = [log(p_trans) + log(1 - trans_red) * (t >= chp) + log(1 - trans_red2) * (t >= chp2) + (t - reversion_time) * log_r * (t >= reversion_time) for t in ts]
        return min.(exp.(log_p), p_trans)
end

ts_risk = 1:365
p_sx_trans_risks = map((p_tr, red_sx_tr, red_sx_tr2, ch1) -> generate_trans_risk_over_time(p_tr, red_sx_tr, red_sx_tr2, ch1, chp_t2, ts_risk),
        prob_trans, red_sx_trans, red_sx_trans2, chp1) .|> x -> reshape(x, length(x), 1)
p_sx_trans_risks_4wk = map((p_tr, red_sx_tr, red_sx_tr2, ch1) -> generate_trans_risk_over_time(p_tr, red_sx_tr, red_sx_tr2, ch1, chp_t2, ts_risk,4*7),
                        prob_trans, red_sx_trans, red_sx_trans2, chp1) .|> x -> reshape(x, length(x), 1)
p_sx_trans_risks_12wk = map((p_tr, red_sx_tr, red_sx_tr2, ch1) -> generate_trans_risk_over_time(p_tr, red_sx_tr, red_sx_tr2, ch1, chp_t2, ts_risk,12*7),
                        prob_trans, red_sx_trans, red_sx_trans2, chp1) .|> x -> reshape(x, length(x), 1)


sx_trans_risk_cred_int = prev_cred_intervals(p_sx_trans_risks)
sx_trans_risk_cred_int_4wk = prev_cred_intervals(p_sx_trans_risks_4wk)
sx_trans_risk_cred_int_12wk = prev_cred_intervals(p_sx_trans_risks_12wk)

dates = [Date(2021, 12, 31) + Day(t) for t in ts_risk]
f = findfirst(dates .== Date(2022, 7, 23))

#Posterior probability of >10% decrease in risk
# p_sx_risk_dec = mean([p_trans[f] < p_trans[1]*0.9 for p_trans in p_sx_trans_risks ])



plt_chng = plot(dates, sx_trans_risk_cred_int.mean_pred,
        ribbon=(sx_trans_risk_cred_int.lb_pred_10, sx_trans_risk_cred_int.ub_pred_10),
        lw=3, fillalpha=0.2,
        lab="Transmission probability (No reversion)",
        title="Transmission probability (sexual contacts)",
        ylabel="Prob. per sexual contact",
        xlims=(long_wks[1] - Day(7), long_wks[end] - Day(7)),
        ylims = (0,0.5),
        xticks=([Date(2022, 5, 1) + Month(k) for k = 0:6], [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:6]),
        left_margin=5mm,
        size=(800, 600), dpi=250,
        tickfont=11, titlefont=17, guidefont=18, legendfont=11)

plot!(plt_chng, dates[f:end], sx_trans_risk_cred_int_4wk.mean_pred[f:end],
        ribbon=(sx_trans_risk_cred_int_4wk.lb_pred_10[f:end], sx_trans_risk_cred_int_4wk.ub_pred_10[f:end]),
        lw=3, fillalpha=0.3,
        lab="Transmission probability (4 week reversion)")

plot!(plt_chng, dates[f:end], sx_trans_risk_cred_int_12wk.mean_pred[f:end],
        ribbon=(sx_trans_risk_cred_int_12wk.lb_pred_10[f:end], sx_trans_risk_cred_int_12wk.ub_pred_10[f:end]),
        lw=3, fillalpha=0.3,color = :black,
        lab="Transmission probability (12 week reversion)")


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

p_oth_trans_risks_4wks = map((p_tr, red_sx_tr, red_sx_tr2, ch1) -> generate_trans_risk_over_time(p_tr, red_sx_tr, red_sx_tr2, ch1, chp_t2, ts_risk, 4*7),
        R0_other, red_oth_trans, red_oth_trans2, chp1) .|> x -> reshape(x, length(x), 1)

p_oth_trans_risks_12wks = map((p_tr, red_sx_tr, red_sx_tr2, ch1) -> generate_trans_risk_over_time(p_tr, red_sx_tr, red_sx_tr2, ch1, chp_t2, ts_risk, 12*7),
        R0_other, red_oth_trans, red_oth_trans2, chp1) .|> x -> reshape(x, length(x), 1)

oth_sx_trans_risk_cred_int = prev_cred_intervals(p_oth_trans_risks)
oth_sx_trans_risk_cred_int4wks = prev_cred_intervals(p_oth_trans_risks_4wks)
oth_sx_trans_risk_cred_int12wks = prev_cred_intervals(p_oth_trans_risks_12wks)

#Posterior probability of >10% decrease in risk
# p_oth_risk_dec = mean([p_trans[f] < p_trans[1]*0.9 for p_trans in p_oth_trans_risks ])


plt_chng_oth = plot(dates, oth_sx_trans_risk_cred_int.mean_pred,
        ribbon=(oth_sx_trans_risk_cred_int.lb_pred_10, oth_sx_trans_risk_cred_int.ub_pred_10),
        lw=3, fillalpha=0.2,
        lab="R0, other contacts(no reversion)",
        title="Reproductive number (other contacts)",
        ylabel="R(t) (non-sexual contacts)",
        xlims=(long_wks[1] - Day(7), long_wks[end] - Day(7)),
        # ylims = (0,0.65),
        xticks=([Date(2022, 5, 1) + Month(k) for k = 0:6], [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:6]),
        left_margin=5mm,
        size=(800, 600), dpi=250,
        tickfont=11, titlefont=17, guidefont=18, legendfont=11)

plot!(plt_chng_oth, dates[f:end], oth_sx_trans_risk_cred_int4wks.mean_pred[f:end],
        ribbon=(oth_sx_trans_risk_cred_int4wks.lb_pred_10[f:end], oth_sx_trans_risk_cred_int4wks.ub_pred_10[f:end]),
        lw=3, fillalpha=0.3,
        lab="R0, other (4 week reversion)")

plot!(plt_chng_oth, dates[f:end], oth_sx_trans_risk_cred_int12wks.mean_pred[f:end],
        ribbon=(oth_sx_trans_risk_cred_int12wks.lb_pred_10[f:end], oth_sx_trans_risk_cred_int12wks.ub_pred_10[f:end]),
        lw=3, fillalpha=0.3,color = :black,
        lab="R0, other (12 week reversion)")

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
prev_cred_int_4wkrev = prev_cred_intervals([pred[3] for pred in preds_and_incidence_interventions_4wkrev])
prev_cred_int_12wkrev = prev_cred_intervals([pred[3] for pred in preds_and_incidence_interventions_12wkrev])

prev_cred_int_cvac = prev_cred_intervals([pred[3] for pred in preds_and_incidence_interventions_cvac])
prev_cred_int_cvac_4wkrev = prev_cred_intervals([pred[3] for pred in preds_and_incidence_interventions_cvac_4wkrev])
prev_cred_int_cvac_12wkrev = prev_cred_intervals([pred[3] for pred in preds_and_incidence_interventions_cvac_12wkrev])

prev_cred_int_overall = prev_cred_intervals([sum(pred[3][:, 1:10], dims=2) for pred in preds_and_incidence_interventions])
prev_cred_int_overall_4wkrev = prev_cred_intervals([sum(pred[3][:, 1:10], dims=2) for pred in preds_and_incidence_interventions_4wkrev])
prev_cred_int_overall_cvac_4wkrev = prev_cred_intervals([sum(pred[3][:, 1:10], dims=2) for pred in preds_and_incidence_interventions_cvac_4wkrev])


##

N_msm_grp = N_msm .* ps'
_wks = long_wks .- Day(7)

plt_prev = plot(_wks[3:end], prev_cred_int_12wkrev.mean_pred[3:end, 1:10] ./ N_msm_grp,
        ribbon=(prev_cred_int_12wkrev.lb_pred_10[3:end, 1:10] ./ N_msm_grp, prev_cred_int_12wkrev.ub_pred_10[3:end, 1:10] ./ N_msm_grp),
        lw=[j / 1.5 for i = 1:1, j = 1:10],
        lab=hcat(["12 week reversion"], fill("", 1, 9)),
        yticks=(0:0.04:0.26, [string(round(y * 100)) * "%" for y in 0:0.04:0.26]),
        xticks=([Date(2022, 5, 1) + Month(k) for k = 0:6], [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:6]),
        ylabel="Prevalence",
        title="MPX prevalence by sexual activity group (MSM)",
        color=:black,
        fillalpha=0.1,
        left_margin=5mm,
        size=(800, 600), dpi=250,
        tickfont=11, titlefont=18, guidefont=18, legendfont=11)

plot!(plt_prev, _wks[(d_proj+1):end], prev_cred_int_cvac_12wkrev.mean_pred[(d_proj+1):end, 1:10] ./ N_msm_grp,
        lw=[j / 1.5 for i = 1:1, j = 1:10],
        ls=:dash,
        color=:black,
        lab=hcat(["12 week reversion (vaccination ceases)"], fill("", 1, 9)))

plot!(plt_prev, _wks[(d_proj+1):end], prev_cred_int_4wkrev.mean_pred[(d_proj+1):end, 1:10] ./ N_msm_grp,
        lw=[j / 1.5 for i = 1:1, j = 1:10],
        ribbon=(prev_cred_int_4wkrev.lb_pred_10[(d1+1):end, 1:10] ./ N_msm_grp, prev_cred_int_4wkrev.ub_pred_10[(d1+1):end, 1:10] ./ N_msm_grp),
        fillalpha = 0.1,
        color=2,
        lab=hcat(["4 week reversion"], fill("", 1, 9)))

plot!(plt_prev, _wks[(d_proj+1):end], prev_cred_int_cvac_4wkrev.mean_pred[(d_proj+1):end, 1:10] ./ N_msm_grp,
        lw=[j / 1.5 for i = 1:1, j = 1:10],
        color=2, ls = :dash,
        lab=hcat(["4 week reversion (ceased vaccination)"], fill("", 1, 9)))

plot!(plt_prev, _wks[3:end], prev_cred_int_overall_4wkrev.mean_pred[3:end,:] ./ N_msm,
        ribbon=(prev_cred_int_overall_4wkrev.lb_pred_10[3:end] ./ N_msm, prev_cred_int_overall_4wkrev.ub_pred_10[3:end] ./ N_msm),
        lw=3,
        color=1,
        fillalpha=0.1,
        lab="",
        xticks=([Date(2022, 5, 1) + Month(k) for k = 0:6], [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:6]),
        yticks=(0:0.001:0.0050, [string(y * 100) * "%" for y in 0:0.001:0.0050]),
        ylims = (0.,0.005),
        inset=bbox(0.7, 0.355, 0.25, 0.25, :top, :left),
        xtickfont=7,
        subplot=2,
        grid=nothing,
        title="Overall: 4 weeks rev.")        

plot!(plt_prev, _wks[3:end], prev_cred_int_overall_cvac_4wkrev.mean_pred[3:end] ./ N_msm,
        subplot=2,
        color=1, ls=:dash, lw=3, lab="")        

##
plt_prev_overall = plot(_wks[3:end], prev_cred_int_12wkrev.mean_pred[3:end, 11] ./ (N_uk - N_msm),
        ribbon=(prev_cred_int_12wkrev.lb_pred_10[3:end, 11] ./ (N_uk - N_msm), prev_cred_int_12wkrev.ub_pred_10[3:end, 11] ./ (N_uk - N_msm)),
        lw=3,
        lab="12 week reversion",
        ylabel="Prevalence",
        title="MPX Prevalence (non-GBMSM)",
        color=:black,
        yticks=(0:1e-6:7.0e-6, [string(round(y * 100, digits=10)) * "%" for y in 0:1e-6:7.0e-6]),
        xticks=([Date(2022, 5, 1) + Month(k) for k = 0:6], [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:6]),
        fillalpha=0.2,
        left_margin=5mm,
        size=(800, 600), dpi=250,
        tickfont=11, titlefont=18, guidefont=18, legendfont=11)

plot!(plt_prev_overall, _wks[(d_proj+1):end], prev_cred_int_cvac_12wkrev.mean_pred[(d_proj+1):end, 11] ./ (N_uk - N_msm),
        color=:black, 
        ls=:dash, 
        lw=3, 
        lab="12 week reversion (ceased vaccination)")

plot!(plt_prev_overall, _wks[(d_proj+1):end], prev_cred_int_4wkrev.mean_pred[(d_proj+1):end, 11] ./ (N_uk - N_msm),
        ribbon=(prev_cred_int_4wkrev.lb_pred_10[(d_proj+1):end, 11] ./ (N_uk - N_msm), prev_cred_int_4wkrev.ub_pred_10[(d_proj+1):end, 11] ./ (N_uk - N_msm)),
        color=2, 
        fillalpha=0.2,
        lw=3, 
        lab="4 week reversion")        

plot!(plt_prev_overall, _wks[(d_proj+1):end], prev_cred_int_cvac_4wkrev.mean_pred[(d_proj+1):end, 11] ./ (N_uk - N_msm),
        color=2, ls=:dash, lw=3, lab="4 week reversion (ceased vaccination)")        

##

plt = plot(plt_chng, plt_chng_oth, plt_prev, plt_prev_overall,
        size=(1750, 1600), dpi=250,
        left_margin=10mm,
        bottom_margin=10mm,
        right_margin=10mm,
        layout=(2, 2))
display(plt)
savefig(plt, "plots/change_and_prevalence" * string(wks[end]) * ".png")

