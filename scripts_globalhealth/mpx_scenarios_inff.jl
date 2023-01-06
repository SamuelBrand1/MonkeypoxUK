using Distributions, StatsBase, StatsPlots
using Plots.PlotMeasures
using LinearAlgebra, RecursiveArrayTools
using OrdinaryDiffEq, ApproxBayes, MCMCChains
using JLD2
using MonkeypoxUK

## Grab UK data and model set up
include("mpxv_datawrangling_inff.jl");
include("setup_model.jl");

##Load posterior draws
# param_draws1 = load("posteriors/posterior_param_draws_2022-08-26_vs1.jld2")["param_draws"]
# param_draws2 = load("posteriors/posterior_param_draws_2022-08-26_vs2.jld2")["param_draws"]
param_draws = mapreduce(
    fn -> load(fn)["param_draws"],
    vcat,
    [
        "posteriors/posterior_param_draws_2022-08-26_vs1.jld2",
        "posteriors/posterior_param_draws_2022-08-26_vs2.jld2",
    ],
)

## Public health emergency effect forecasts
long_wks = [wks; [wks[end] + Day(7 * k) for k = 1:12]]
long_mpxv_wkly = [mpxv_wkly; zeros(12, 2)]
# wkly_vaccinations = [zeros(12); 1000; 2000; fill(5000, 13)] * 1.5

plt_vacs = plot(
    [wks[1] + Day(7 * (k - 1)) for k = 1:size(wkly_vaccinations, 1)],
    cumsum(wkly_vaccinations),
    title = "Cumulative number of MPX vaccine doses",
    lab = "Projection",
    color = :black,
    lw = 3,# yticks=0:1000:8000,
    ylabel = "Cum. doses",
    size = (800, 600),
    left_margin = 5mm,
    guidefont = 16,
    tickfont = 11,
    titlefont = 18,
    legend = :topleft,
    right_margin = 5mm,
)
scatter!(plt_vacs, [Date(2022, 8, 30)], [38_079], lab = "UKHSA reporting", ms = 8)
display(plt_vacs)
savefig(plt_vacs, "plots/vaccine_rollout.png")


##
chp_t2 = (Date(2022, 7, 23) - Date(2021, 12, 31)).value #Announcement of Public health emergency
inf_duration_red = 0.0


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

no_interventions_ensemble = [
    (
        trans_red2 = 0.0,
        vac_effectiveness = 0.0,
        trans_red_other2 = 0.0,
        wkly_vaccinations = zeros(size(long_mpxv_wkly)),
        chp_t2,
        inf_duration_red,
    ) for θ in param_draws
]

no_red_ensemble = [
    (
        trans_red2 = 0.0,
        vac_effectiveness = rand(Uniform(0.7, 0.85)),
        trans_red_other2 = 0.0,
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
        wkly_vaccinations = zeros(size(long_mpxv_wkly)),
        chp_t2,
        inf_duration_red,
    ) for θ in param_draws
]

mpx_sim_function_interventions = MonkeypoxUK.mpx_sim_function_interventions
preds_and_incidence_interventions = map(
    (θ, intervention) ->
        mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention)[2:4],
    param_draws,
    interventions_ensemble,
)
preds_and_incidence_no_interventions = map(
    (θ, intervention) ->
        mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention)[2:4],
    param_draws,
    no_interventions_ensemble,
)
preds_and_incidence_no_vaccines = map(
    (θ, intervention) ->
        mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention)[2:4],
    param_draws,
    no_vac_ensemble,
)
preds_and_incidence_no_redtrans = map(
    (θ, intervention) ->
        mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention)[2:4],
    param_draws,
    no_red_ensemble,
)

## Cumulative incidence on week 15 (starting 8th August) for highest frequency group for paper
cum_inc_wk_15 = [
    sum(pred[2][1:15, 10]) ./ (N_msm * ps[10]) for pred in preds_and_incidence_interventions
]
mean(cum_inc_wk_15)
quantile(cum_inc_wk_15, [0.25, 0.5, 0.75])

##Gather data
d1, d2 = size(mpxv_wkly)

preds = [x[1] for x in preds_and_incidence_interventions]
preds_nointervention = [x[1] for x in preds_and_incidence_no_interventions]
preds_novacs = [x[1] for x in preds_and_incidence_no_vaccines]
preds_noredtrans = [x[1] for x in preds_and_incidence_no_redtrans]

cum_cases_forwards =
    [cumsum(x[1][(d1+1):end, :], dims = 1) for x in preds_and_incidence_interventions]
cum_cases_nointervention_forwards =
    [cumsum(x[1][(d1+1):end, :], dims = 1) for x in preds_and_incidence_no_interventions]
cum_cases_novaccines_forwards =
    [cumsum(x[1][(d1+1):end, :], dims = 1) for x in preds_and_incidence_no_vaccines]
cum_cases_noredtrans_forwards =
    [cumsum(x[1][(d1+1):end, :], dims = 1) for x in preds_and_incidence_no_redtrans]


##Simulation projections
long_wks = [wks; [wks[end] + Day(7 * k) for k = 1:12]]
long_mpxv_wkly = [mpxv_wkly; zeros(12, 2)]

cred_int = MonkeypoxUK.cred_intervals(preds)
cred_int_rwc = MonkeypoxUK.cred_intervals(preds_nointervention)
cred_int_nv = MonkeypoxUK.cred_intervals(preds_novacs)
cred_int_nr = MonkeypoxUK.cred_intervals(preds_noredtrans)

## MSM projections
plt_msm = plot(;
    ylabel = "Weekly cases",
    title = "UK Monkeypox Case Projections (MSM)",# yscale=:log10,
    legend = :topleft,
    # yticks=([1, 2, 11, 101, 1001], [0, 1, 10, 100, 1000]),
    ylims = (-5, 700),
    xticks = (
        [Date(2022, 5, 1) + Month(k) for k = 0:6],
        [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:6],
    ),
    left_margin = 5mm,
    size = (800, 600),
    dpi = 250,
    tickfont = 11,
    titlefont = 18,
    guidefont = 18,
    legendfont = 11,
)
plot!(
    plt_msm,
    long_wks,
    cred_int.mean_pred[:, 1],
    ribbon = (cred_int.lb_pred_10[:, 1], cred_int.ub_pred_10[:, 1]),
    lw = 3,
    color = :black,
    fillalpha = 0.2,
    lab = "Forecast",
)
plot!(
    plt_msm,
    long_wks[11:end],
    cred_int_nr.mean_pred[11:end, 1],
    ribbon = (cred_int_nr.lb_pred_10[11:end, 1], cred_int_nr.ub_pred_10[11:end, 1]),
    lw = 3,
    ls = :dash,
    color = 1,
    fillalpha = 0.2,
    lab = "No further behavioural response",
)
plot!(
    plt_msm,
    long_wks[11:end],
    cred_int_nv.mean_pred[11:end, 1],
    ribbon = (cred_int_nv.lb_pred_10[11:end, 1], cred_int_nv.ub_pred_10[11:end, 1]),
    lw = 3,
    ls = :dash,
    color = 4,
    fillalpha = 0.2,
    lab = "No vaccinations",
)
plot!(
    plt_msm,
    long_wks[11:end],
    cred_int_rwc.mean_pred[11:end, 1],
    ribbon = (cred_int_rwc.lb_pred_10[11:end, 1], cred_int_rwc.ub_pred_10[11:end, 1]),
    lw = 3,
    ls = :dash,
    color = 2,
    fillalpha = 0.2,
    lab = "Reasonable worst case scenario",
)
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
    wks[1:(end-1)],
    mpxv_wkly[1:(end-1), 1],
    lab = "Data",
    ms = 6,
    color = :black,
)

##
plt_nmsm = plot(;
    ylabel = "Weekly cases",
    title = "UK Monkeypox Case Projections (non-MSM)",# yscale=:log10,
    legend = :topleft,
    # yticks=([1, 2, 11, 101, 1001], [0, 1, 10, 100, 1000]),
    ylims = (-1, 50),
    xticks = (
        [Date(2022, 5, 1) + Month(k) for k = 0:6],
        [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:6],
    ),
    left_margin = 5mm,
    size = (800, 600),
    dpi = 250,
    tickfont = 11,
    titlefont = 18,
    guidefont = 18,
    legendfont = 11,
)
plot!(
    plt_nmsm,
    long_wks,
    cred_int.mean_pred[:, 2],
    ribbon = (cred_int.lb_pred_10[:, 2], cred_int.ub_pred_10[:, 2]),
    lw = 3,
    color = :black,
    fillalpha = 0.2,
    lab = "Forecast",
)
plot!(
    plt_nmsm,
    long_wks[11:end],
    cred_int_nr.mean_pred[11:end, 2],
    ribbon = (cred_int_nr.lb_pred_10[11:end, 2], cred_int_nr.ub_pred_10[11:end, 2]),
    lw = 3,
    ls = :dash,
    color = 1,
    fillalpha = 0.2,
    lab = "No further behavioural response",
)
plot!(
    plt_nmsm,
    long_wks[11:end],
    cred_int_nv.mean_pred[11:end, 2],
    ribbon = (cred_int_nv.lb_pred_10[11:end, 2], cred_int_nv.ub_pred_10[11:end, 2]),
    lw = 3,
    ls = :dash,
    color = 4,
    fillalpha = 0.2,
    lab = "No vaccinations",
)
plot!(
    plt_nmsm,
    long_wks[11:end],
    cred_int_rwc.mean_pred[11:end, 2],
    ribbon = (cred_int_rwc.lb_pred_10[11:end, 2], cred_int_rwc.ub_pred_10[11:end, 2]),
    lw = 3,
    ls = :dash,
    color = 2,
    fillalpha = 0.2,
    lab = "Reasonable worst case scenario",
)
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
    wks[1:(end-1)],
    mpxv_wkly[1:(end-1), 2],
    lab = "Data",
    ms = 6,
    color = :black,
)



##cumulative Incidence plots
# cred_int_cum_incidence = cred_intervals(cum_incidences)
# cred_int_cum_incidence_no_intervention = cred_intervals(cum_incidences_nointervention)
cred_int_cum_incidence = MonkeypoxUK.cred_intervals(cum_cases_forwards)
cred_int_cum_incidence_no_intervention =
    MonkeypoxUK.cred_intervals(cum_cases_nointervention_forwards)
cred_int_cum_no_vaccines = MonkeypoxUK.cred_intervals(cum_cases_novaccines_forwards)
cred_int_cum_noredtrans = MonkeypoxUK.cred_intervals(cum_cases_noredtrans_forwards)


total_cases = sum(mpxv_wkly, dims = 1)
plt_cm_msm = plot(;
    ylabel = "Cumulative cases",
    title = "UK Monkeypox cumulative case projections (MSM)",#yscale=:log10,
    legend = :topleft,
    # yticks=(0:2500:12500, 0:2500:100),
    xticks = (
        [Date(2022, 5, 1) + Month(k) for k = 0:6],
        [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:6],
    ),
    left_margin = 5mm,
    size = (800, 600),
    dpi = 250,
    tickfont = 11,
    titlefont = 18,
    guidefont = 18,
    legendfont = 11,
)
plot!(
    plt_cm_msm,
    long_wks[((d1+1)):end],
    total_cases[:, 1] .+ cred_int_cum_incidence.mean_pred[:, 1],
    ribbon = (
        cred_int_cum_incidence.lb_pred_10[:, 1],
        cred_int_cum_incidence.ub_pred_10[:, 1],
    ),
    lw = 3,
    color = :black,
    fillalpha = 0.2,
    lab = "Forecast",
)

plot!(
    plt_cm_msm,
    long_wks[((d1+1)):end],
    total_cases[:, 1] .+ cred_int_cum_noredtrans.mean_pred[:, 1],
    ribbon = (
        cred_int_cum_noredtrans.lb_pred_10[:, 1],
        cred_int_cum_noredtrans.ub_pred_10[:, 1],
    ),
    lw = 3,
    ls = :dash,
    color = 1,
    fillalpha = 0.2,
    lab = "No further behavioural response",
)
plot!(
    plt_cm_msm,
    long_wks[((d1+1)):end],
    total_cases[:, 1] .+ cred_int_cum_no_vaccines.mean_pred[:, 1],
    ribbon = (
        cred_int_cum_no_vaccines.lb_pred_10[:, 1],
        cred_int_cum_no_vaccines.ub_pred_10[:, 1],
    ),
    lw = 3,
    ls = :dash,
    color = 4,
    fillalpha = 0.2,
    lab = "No vaccinations",
)

plot!(
    plt_cm_msm,
    long_wks[((d1+1)):end],
    total_cases[:, 1] .+ cred_int_cum_incidence_no_intervention.mean_pred[:, 1],
    ribbon = (
        cred_int_cum_incidence_no_intervention.lb_pred_10[:, 1],
        cred_int_cum_incidence_no_intervention.ub_pred_10[:, 1],
    ),
    lw = 3,
    ls = :dash,
    color = 2,
    fillalpha = 0.2,
    lab = "Reasonable worst case scenario",
)

scatter!(
    plt_cm_msm,
    wks,
    cumsum(mpxv_wkly[:, 1], dims = 1),
    lab = "Data",
    ms = 6,
    color = :black,
)

##
total_cases = sum(mpxv_wkly, dims = 1)
plt_cm_nmsm = plot(;
    ylabel = "Cumulative cases",
    title = "UK Monkeypox cumulative case projections (non-MSM)",#yscale=:log10,
    legend = :topleft,
    # yticks=(0:2500:12500, 0:2500:12500),
    xticks = (
        [Date(2022, 5, 1) + Month(k) for k = 0:6],
        [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:6],
    ),
    left_margin = 5mm,
    size = (800, 600),
    dpi = 250,
    tickfont = 11,
    titlefont = 17,
    guidefont = 18,
    legendfont = 11,
)
plot!(
    plt_cm_nmsm,
    long_wks[((d1+1)):end],
    total_cases[:, 2] .+ cred_int_cum_incidence.mean_pred[:, 2],
    ribbon = (
        cred_int_cum_incidence.lb_pred_10[:, 2],
        cred_int_cum_incidence.ub_pred_10[:, 2],
    ),
    lw = 3,
    color = :black,
    fillalpha = 0.2,
    lab = "Forecast",
)

plot!(
    plt_cm_nmsm,
    long_wks[((d1+1)):end],
    total_cases[:, 2] .+ cred_int_cum_noredtrans.mean_pred[:, 2],
    ribbon = (
        cred_int_cum_noredtrans.lb_pred_10[:, 2],
        cred_int_cum_noredtrans.ub_pred_10[:, 2],
    ),
    lw = 3,
    ls = :dash,
    color = 1,
    fillalpha = 0.2,
    lab = "No further behavioural response",
)
plot!(
    plt_cm_nmsm,
    long_wks[((d1+1)):end],
    total_cases[:, 2] .+ cred_int_cum_no_vaccines.mean_pred[:, 2],
    ribbon = (
        cred_int_cum_no_vaccines.lb_pred_10[:, 2],
        cred_int_cum_no_vaccines.ub_pred_10[:, 2],
    ),
    lw = 3,
    ls = :dash,
    color = 4,
    fillalpha = 0.2,
    lab = "No vaccinations",
)

plot!(
    plt_cm_nmsm,
    long_wks[((d1+1)):end],
    total_cases[:, 2] .+ cred_int_cum_incidence_no_intervention.mean_pred[:, 2],
    ribbon = (
        cred_int_cum_incidence_no_intervention.lb_pred_10[:, 2],
        cred_int_cum_incidence_no_intervention.ub_pred_10[:, 2],
    ),
    lw = 3,
    ls = :dash,
    color = 2,
    fillalpha = 0.2,
    lab = "Reasonable worst case scenario",
)

scatter!(
    plt_cm_nmsm,
    wks,
    cumsum(mpxv_wkly[:, 2], dims = 1),
    lab = "Data",
    ms = 6,
    color = :black,
)

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
# savefig(plt, "plots/case_projections_ukhsa.png")


## Change in transmission over time

prob_trans = [θ[4] for θ in param_draws]
red_sx_trans = [θ[9] for θ in param_draws]
chp1 = [θ[8] for θ in param_draws]
red_sx_trans2 = [int.trans_red2 for int in interventions_ensemble]

function generate_trans_risk_over_time(p_trans, trans_red, trans_red2, chp, chp2, ts)
    log_p = [
        log(p_trans) + log(1 - trans_red) * (t >= chp) + log(1 - trans_red2) * (t >= chp2) for t in ts
    ]
    return exp.(log_p)
end
ts_risk = 1:365
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
p_sx_trans_risks_rwc =
    map(
        (p_tr, red_sx_tr, red_sx_tr2, ch1) ->
            generate_trans_risk_over_time(p_tr, red_sx_tr, red_sx_tr2, ch1, Inf, ts_risk),
        prob_trans,
        red_sx_trans,
        red_sx_trans2,
        chp1,
    ) .|> x -> reshape(x, length(x), 1)


sx_trans_risk_cred_int = prev_cred_intervals(p_sx_trans_risks)
sx_trans_risk_cred_no_int = prev_cred_intervals(p_sx_trans_risks_rwc)

dates = [Date(2021, 12, 31) + Day(t) for t in ts_risk]
f = findfirst(dates .== Date(2022, 7, 23))

#Posterior probability of >10% decrease in risk
p_sx_risk_dec = mean([p_trans[f] < p_trans[1] * 0.9 for p_trans in p_sx_trans_risks])



plt_chng = plot(
    dates,
    sx_trans_risk_cred_int.mean_pred,
    ribbon = (sx_trans_risk_cred_int.lb_pred_25, sx_trans_risk_cred_int.ub_pred_25),
    lw = 3,
    fillalpha = 0.2,
    lab = "Transmission probability (forecast)",
    title = "Transmission probability (sexual contacts)",
    ylabel = "Prob. per sexual contact",
    xlims = (long_wks[1] - Day(7), long_wks[end] - Day(7)),
    ylims = (0, 0.4),
    xticks = (
        [Date(2022, 5, 1) + Month(k) for k = 0:6],
        [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:6],
    ),
    left_margin = 5mm,
    size = (800, 600),
    dpi = 250,
    tickfont = 11,
    titlefont = 17,
    guidefont = 18,
    legendfont = 11,
)

plot!(
    plt_chng,
    dates[f:end],
    sx_trans_risk_cred_no_int.mean_pred[f:end],
    ribbon = (
        sx_trans_risk_cred_no_int.lb_pred_25[f:end],
        sx_trans_risk_cred_no_int.ub_pred_25[f:end],
    ),
    lw = 3,
    fillalpha = 0.3,
    lab = "Transmission probability (no change)",
)
vline!(
    plt_chng,
    [Date(2022, 7, 23)],
    lw = 4,
    color = :black,
    ls = :dash,
    lab = "WHO declaration",
)
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
p_oth_trans_risks_rwc =
    map(
        (p_tr, red_sx_tr, red_sx_tr2, ch1) ->
            generate_trans_risk_over_time(p_tr, red_sx_tr, red_sx_tr2, ch1, Inf, ts_risk),
        R0_other,
        red_oth_trans,
        red_oth_trans2,
        chp1,
    ) .|> x -> reshape(x, length(x), 1)

oth_sx_trans_risk_cred_int = prev_cred_intervals(p_oth_trans_risks)
oth_trans_risk_cred_no_int = prev_cred_intervals(p_oth_trans_risks_rwc)

#Posterior probability of >10% decrease in risk
p_oth_risk_dec = mean([p_trans[f] < p_trans[1] * 0.9 for p_trans in p_oth_trans_risks])


plt_chng_oth = plot(
    dates,
    oth_sx_trans_risk_cred_int.mean_pred,
    ribbon = (oth_sx_trans_risk_cred_int.lb_pred_25, oth_sx_trans_risk_cred_int.ub_pred_25),
    lw = 3,
    fillalpha = 0.2,
    lab = "R0, non-sexual (forecast)",
    title = "Reproductive number (non-sexual contacts)",
    ylabel = "R(t) (non-sexual contacts)",
    xlims = (long_wks[1] - Day(7), long_wks[end] - Day(7)),
    ylims = (0, 0.65),
    xticks = (
        [Date(2022, 5, 1) + Month(k) for k = 0:6],
        [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:6],
    ),
    left_margin = 5mm,
    size = (800, 600),
    dpi = 250,
    tickfont = 11,
    titlefont = 17,
    guidefont = 18,
    legendfont = 11,
)

plot!(
    plt_chng_oth,
    dates[f:end],
    oth_trans_risk_cred_no_int.mean_pred[f:end],
    ribbon = (
        oth_trans_risk_cred_no_int.lb_pred_25[f:end],
        oth_trans_risk_cred_no_int.ub_pred_25[f:end],
    ),
    lw = 3,
    fillalpha = 0.3,
    lab = "R0, non-sexual (no change)",
)
vline!(
    plt_chng_oth,
    [Date(2022, 7, 23)],
    lw = 4,
    color = :black,
    ls = :dash,
    lab = "WHO declaration",
)
## Combined plot

plt = plot(
    plt_chng,
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


##Prevalence plots

prev_cred_int = prev_cred_intervals([pred[3] for pred in preds_and_incidence_interventions])
prev_cred_no_int =
    prev_cred_intervals([pred[3] for pred in preds_and_incidence_no_interventions])
prev_cred_int_overall = prev_cred_intervals([
    sum(pred[3][:, 1:10], dims = 2) for pred in preds_and_incidence_interventions
])
prev_cred_no_int_overall = prev_cred_intervals([
    sum(pred[3][:, 1:10], dims = 2) for pred in preds_and_incidence_no_interventions
])


N_msm_grp = N_msm .* ps'
_wks = long_wks .- Day(7)
plt_prev = plot(
    _wks,
    prev_cred_int.mean_pred[:, 1:10] ./ N_msm_grp,
    ribbon = (
        prev_cred_int.lb_pred_25[:, 1:10] ./ N_msm_grp,
        prev_cred_int.ub_pred_25[:, 1:10] ./ N_msm_grp,
    ),
    lw = [j / 1.5 for i = 1:1, j = 1:10],
    lab = hcat(["Forecast"], fill("", 1, 9)),
    yticks = (0:0.02:0.18, [string(round(y * 100)) * "%" for y = 0:0.02:0.18]),
    xticks = (
        [Date(2022, 5, 1) + Month(k) for k = 0:6],
        [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:6],
    ),
    ylabel = "Prevalence",
    title = "MPX prevalence by sexual activity group (MSM)",
    color = :black,
    fillalpha = 0.1,
    left_margin = 5mm,
    size = (800, 600),
    dpi = 250,
    tickfont = 11,
    titlefont = 18,
    guidefont = 18,
    legendfont = 11,
)

plot!(
    plt_prev,
    _wks[11:end],
    prev_cred_no_int.mean_pred[11:end, 1:10] ./ N_msm_grp,
    lw = [j / 1.5 for i = 1:1, j = 1:10],
    ls = :dash,
    color = :black,
    lab = hcat(["Worst case scenario"], fill("", 1, 9)),
)
plot!(
    plt_prev,
    _wks,
    prev_cred_int_overall.mean_pred ./ N_msm,
    ribbon = (
        prev_cred_int_overall.lb_pred_25 ./ N_msm,
        prev_cred_int_overall.ub_pred_25 ./ N_msm,
    ),
    lw = 3,
    color = 1,
    fillalpha = 0.1,
    lab = "",
    xticks = (
        [Date(2022, 5, 1) + Month(k) for k = 0:5],
        [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:5],
    ),
    yticks = (0:0.001:0.0050, [string(y * 100) * "%" for y = 0:0.001:0.0040]),
    inset = bbox(0.7, 0.255, 0.25, 0.25, :top, :left),
    xtickfont = 7,
    subplot = 2,
    grid = nothing,
    title = "Overall",
)
plot!(
    plt_prev,
    _wks,
    prev_cred_no_int_overall.mean_pred ./ N_msm,
    subplot = 2,
    color = 1,
    ls = :dash,
    lw = 3,
    lab = "",
)
##
plt_prev_overall = plot(
    _wks,
    prev_cred_int.mean_pred[:, 11] ./ (N_uk - N_msm),
    ribbon = (
        prev_cred_int.lb_pred_25[:, 11] ./ (N_uk - N_msm),
        prev_cred_int.ub_pred_25[:, 11] ./ (N_uk - N_msm),
    ),
    lw = 3,
    lab = "Forecast",
    ylabel = "Prevalence",
    title = "MPX Prevalence (non-MSM)",
    color = 2,
    yticks = (
        0:5e-7:4.0e-6,
        [string(round(y * 100, digits = 10)) * "%" for y = 0:5e-7:4.0e-6],
    ),
    xticks = (
        [Date(2022, 5, 1) + Month(k) for k = 0:6],
        [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:6],
    ),
    fillalpha = 0.2,
    left_margin = 5mm,
    size = (800, 600),
    dpi = 250,
    tickfont = 11,
    titlefont = 18,
    guidefont = 18,
    legendfont = 11,
)
plot!(
    plt_prev_overall,
    _wks[11:end],
    prev_cred_no_int.mean_pred[11:end, 11] ./ (N_uk - N_msm),
    color = 2,
    ls = :dash,
    lw = 3,
    lab = "Worst case scenario",
)
##
plt = plot(
    plt_chng,
    plt_chng_oth,
    plt_prev,
    plt_prev_overall,
    size = (1750, 1600),
    dpi = 250,
    left_margin = 10mm,
    bottom_margin = 10mm,
    right_margin = 10mm,
    layout = (2, 2),
)
display(plt)
savefig(plt, "plots/change_and_prevalence" * string(wks[end]) * ".png")
