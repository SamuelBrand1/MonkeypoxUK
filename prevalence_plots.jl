## Only run after running mpx_scenarios.jl script


##Prevalence plots

prev_cred_int = prev_cred_intervals([pred[3] for pred in preds_and_incidence_interventions])
prev_cred_int_4wkrev =
    prev_cred_intervals([pred[3] for pred in preds_and_incidence_interventions_4wkrev])
prev_cred_int_12wkrev =
    prev_cred_intervals([pred[3] for pred in preds_and_incidence_interventions_12wkrev])

prev_cred_int_cvac =
    prev_cred_intervals([pred[3] for pred in preds_and_incidence_interventions_cvac])
prev_cred_int_cvac_4wkrev =
    prev_cred_intervals([pred[3] for pred in preds_and_incidence_interventions_cvac_4wkrev])
prev_cred_int_cvac_12wkrev = prev_cred_intervals([
    pred[3] for pred in preds_and_incidence_interventions_cvac_12wkrev
])

prev_cred_int_overall = prev_cred_intervals([
    sum(pred[3][:, 1:10], dims=2) for pred in preds_and_incidence_interventions
])
prev_cred_int_overall_4wkrev = prev_cred_intervals([
    sum(pred[3][:, 1:10], dims=2) for pred in preds_and_incidence_interventions_4wkrev
])

prev_cred_int_overall_12wkrev = prev_cred_intervals([
    sum(pred[3][:, 1:10], dims=2) for pred in preds_and_incidence_interventions_12wkrev
])

prev_cred_int_overall_cvac_4wkrev = prev_cred_intervals([
    sum(pred[3][:, 1:10], dims=2) for
    pred in preds_and_incidence_interventions_cvac_4wkrev
])


##

N_msm_grp = N_msm .* ps'
_wks = long_wks .- Day(7)


prevalence = DataFrame()
prevalence[:, "Dates"] = _wks[3:end]



plt_prev = plot(
    _wks[3:end],
    prev_cred_int_12wkrev.mean_pred[3:end, 1:10] ./ N_msm_grp,
    ribbon=(
        prev_cred_int_12wkrev.lb_pred_10[3:end, 1:10] ./ N_msm_grp,
        prev_cred_int_12wkrev.ub_pred_10[3:end, 1:10] ./ N_msm_grp,
    ),
    lw=[j / 1.5 for i = 1:1, j = 1:10],
    lab=hcat(["12 week reversion"], fill("", 1, 9)),
    yticks=(0:0.04:0.36, [string(round(y * 100)) * "%" for y = 0:0.04:0.36]),
    xticks=(
        [Date(2022, 5, 1) + Month(k) for k = 0:10],
        [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:10],
    ),
    ylabel="Prevalence",
    title="MPX prevalence by sexual activity group (GBMSM)",
    color=:black,
    fillalpha=0.1,
    left_margin=5mm,
    size=(800, 600),
    dpi=250,
    tickfont=13,
    titlefont=18,
    guidefont=18,
    legendfont=11,
)

for grp = 1:10
    prevalence[:, "Prev. % GBMSM sx activity grp $(grp) (post. mean; 12wk reversion)"] = 100 .* prev_cred_int_12wkrev.mean_pred[3:end, grp] ./ N_msm_grp[grp]
    prevalence[:, "Prev. % GBMSM sx activity grp $(grp) (post. 10%; 12wk reversion)"] = 100 .* (prev_cred_int_12wkrev.mean_pred[3:end, grp] .- prev_cred_int_12wkrev.lb_pred_10[3:end, grp]) ./ N_msm_grp[grp]
    prevalence[:, "Prev. % GBMSM sx activity grp $(grp) (post. 90%; 12wk reversion)"] = 100 .* (prev_cred_int_12wkrev.mean_pred[3:end, grp] .+ prev_cred_int_12wkrev.ub_pred_10[3:end, grp]) ./ N_msm_grp[grp]
end

plot!(
    plt_prev,
    _wks[(d_proj+1):end],
    prev_cred_int_cvac_12wkrev.mean_pred[(d_proj+1):end, 1:10] ./ N_msm_grp,
    lw=[j / 1.5 for i = 1:1, j = 1:10],
    ls=:dash,
    color=:black,
    lab=hcat(["12 week reversion (vaccination ceases)"], fill("", 1, 9)),
)

plot!(
    plt_prev,
    _wks[(d_proj+1):end],
    prev_cred_int_4wkrev.mean_pred[(d_proj+1):end, 1:10] ./ N_msm_grp,
    lw=[j / 1.5 for i = 1:1, j = 1:10],
    ribbon=(
        prev_cred_int_4wkrev.lb_pred_10[(d_proj+1):end, 1:10] ./ N_msm_grp,
        prev_cred_int_4wkrev.ub_pred_10[(d_proj+1):end, 1:10] ./ N_msm_grp,
    ),
    fillalpha=0.1,
    color=2,
    lab=hcat(["4 week reversion"], fill("", 1, 9)),
)

for grp = 1:10
    prevalence[:, "Prev. % GBMSM sx activity grp $(grp) (post. mean; 4wk reversion)"] = 100 .* prev_cred_int_4wkrev.mean_pred[3:end, grp] ./ N_msm_grp[grp]
    prevalence[:, "Prev. % GBMSM sx activity grp $(grp) (post. 10%; 4wk reversion)"] = 100 .* (prev_cred_int_4wkrev.mean_pred[3:end, grp] .- prev_cred_int_4wkrev.lb_pred_10[3:end, grp]) ./ N_msm_grp[grp]
    prevalence[:, "Prev. % GBMSM sx activity grp $(grp) (post. 90%; 4wk reversion)"] = 100 .* (prev_cred_int_4wkrev.mean_pred[3:end, grp] .+ prev_cred_int_4wkrev.ub_pred_10[3:end, grp]) ./ N_msm_grp[grp]
end

plot!(
    plt_prev,
    _wks[(d_proj+1):end],
    prev_cred_int_cvac_4wkrev.mean_pred[(d_proj+1):end, 1:10] ./ N_msm_grp,
    lw=[j / 1.5 for i = 1:1, j = 1:10],
    color=2,
    ls=:dash,
    lab=hcat(["4 week reversion (ceased vaccination)"], fill("", 1, 9)),
)

plot!(
    plt_prev,
    _wks[3:end],
    prev_cred_int_overall_4wkrev.mean_pred[3:end, :] ./ N_msm,
    ribbon=(
        prev_cred_int_overall_4wkrev.lb_pred_10[3:end] ./ N_msm,
        prev_cred_int_overall_4wkrev.ub_pred_10[3:end] ./ N_msm,
    ),
    lw=3,
    color=1,
    fillalpha=0.1,
    lab="",
    xticks=(
        [Date(2022, 5, 1) + Month(k) for k = 0:10],
        [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:10],
    ),
    yticks=(0:0.001:0.0060, [string(y * 100) * "%" for y = 0:0.001:0.0060]),
    ylims=(0.0, 0.007),
    inset=bbox(0.65, 0.355, 0.275, 0.25, :top, :left),
    xtickfont=7,
    subplot=2,
    grid=nothing,
    title="Overall: 4 weeks rev.",
)

prevalence[:, "Prev. % GBMSM overall (post. mean; 4wk reversion)"] = 100 .* prev_cred_int_overall_4wkrev.mean_pred[3:end] ./ N_msm
prevalence[:, "Prev. % GBMSM overall (post. 10%; 4wk reversion)"] = 100 .* (prev_cred_int_overall_4wkrev.mean_pred[3:end] .- prev_cred_int_overall_4wkrev.lb_pred_10[3:end]) ./ N_msm
prevalence[:, "Prev. % GBMSM overall (post. 90%; 4wk reversion)"] = 100 .* (prev_cred_int_overall_4wkrev.mean_pred[3:end] .+ prev_cred_int_overall_4wkrev.ub_pred_10[3:end]) ./ N_msm

prevalence[:, "Prev. % GBMSM overall (post. mean; 12wk reversion)"] = 100 .* prev_cred_int_overall_12wkrev.mean_pred[3:end] ./ N_msm
prevalence[:, "Prev. % GBMSM overall (post. 10%; 12wk reversion)"] = 100 .* (prev_cred_int_overall_12wkrev.mean_pred[3:end] .- prev_cred_int_overall_12wkrev.lb_pred_10[3:end]) ./ N_msm
prevalence[:, "Prev. % GBMSM overall (post. 90%; 12wk reversion)"] = 100 .* (prev_cred_int_overall_12wkrev.mean_pred[3:end] .+ prev_cred_int_overall_12wkrev.ub_pred_10[3:end]) ./ N_msm



plot!(
    plt_prev,
    _wks[3:end],
    prev_cred_int_overall_cvac_4wkrev.mean_pred[3:end] ./ N_msm,
    subplot=2,
    color=1,
    ls=:dash,
    lw=3,
    lab="",
)



##
plt_prev_overall = plot(
    _wks[3:end],
    prev_cred_int_12wkrev.mean_pred[3:end, 11] ./ (N_uk - N_msm),
    ribbon=(
        prev_cred_int_12wkrev.lb_pred_10[3:end, 11] ./ (N_uk - N_msm),
        prev_cred_int_12wkrev.ub_pred_10[3:end, 11] ./ (N_uk - N_msm),
    ),
    lw=3,
    lab="12 week reversion",
    ylabel="Prevalence",
    title="MPX Prevalence (non-GBMSM)",
    color=:black,
    yticks=(
        0:1e-6:10.0e-6,
        [string(round(y * 100, digits=10)) * "%" for y = 0:1e-6:10.0e-6],
    ),
    xticks=(
        [Date(2022, 5, 1) + Month(k) for k = 0:10],
        [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:10],
    ),
    fillalpha=0.2,
    left_margin=5mm,
    size=(800, 600),
    dpi=250,
    tickfont=13,
    titlefont=18,
    guidefont=18,
    legendfont=11,
)

prevalence[:, "Prev. % non-GBMSM overall (post. mean; 4wk reversion)"] = 100 .* prev_cred_int_4wkrev.mean_pred[3:end, 11] ./ (N_uk - N_msm)
prevalence[:, "Prev. % non-GBMSM overall (post. 10%; 4wk reversion)"] = 100 .* (prev_cred_int_4wkrev.mean_pred[3:end, 11] .- prev_cred_int_4wkrev.lb_pred_10[3:end, 11]) ./ (N_uk - N_msm)
prevalence[:, "Prev. % non-GBMSM overall (post. 90%; 4wk reversion)"] = 100 .* (prev_cred_int_4wkrev.mean_pred[3:end, 11] .+ prev_cred_int_4wkrev.ub_pred_10[3:end, 11]) ./ (N_uk - N_msm)

prevalence[:, "Prev. % non-GBMSM overall (post. mean; 12wk reversion)"] = 100 .* prev_cred_int_12wkrev.mean_pred[3:end, 11] ./ (N_uk - N_msm)
prevalence[:, "Prev. % non-GBMSM overall (post. 10%; 12wk reversion)"] = 100 .* (prev_cred_int_12wkrev.mean_pred[3:end, 11] .- prev_cred_int_12wkrev.lb_pred_10[3:end, 11]) ./ (N_uk - N_msm)
prevalence[:, "Prev. % non-GBMSM overall (post. 90%; 12wk reversion)"] = 100 .* (prev_cred_int_12wkrev.mean_pred[3:end, 11] .+ prev_cred_int_12wkrev.ub_pred_10[3:end, 11]) ./ (N_uk - N_msm)


plot!(
    plt_prev_overall,
    _wks[(d_proj+1):end],
    prev_cred_int_cvac_12wkrev.mean_pred[(d_proj+1):end, 11] ./ (N_uk - N_msm),
    color=:black,
    ls=:dash,
    lw=3,
    lab="12 week reversion (ceased vaccination)",
)

plot!(
    plt_prev_overall,
    _wks[(d_proj+1):end],
    prev_cred_int_4wkrev.mean_pred[(d_proj+1):end, 11] ./ (N_uk - N_msm),
    ribbon=(
        prev_cred_int_4wkrev.lb_pred_10[(d_proj+1):end, 11] ./ (N_uk - N_msm),
        prev_cred_int_4wkrev.ub_pred_10[(d_proj+1):end, 11] ./ (N_uk - N_msm),
    ),
    color=2,
    fillalpha=0.2,
    lw=3,
    lab="4 week reversion",
)

plot!(
    plt_prev_overall,
    _wks[(d_proj+1):end],
    prev_cred_int_cvac_4wkrev.mean_pred[(d_proj+1):end, 11] ./ (N_uk - N_msm),
    color=2,
    ls=:dash,
    lw=3,
    lab="4 week reversion (ceased vaccination)",
)


display(plt_prev_overall)

CSV.write("projections/MPX_prev" * string(wks[end]) * ".csv", prevalence)

##