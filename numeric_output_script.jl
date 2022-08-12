## Scratch script for generate forecast numbers for paper
include("mpx_scenarios.jl");

##
#GBMSM peak prev
ovrall_prev_mean = prev_cred_int_overall.mean_pred ./ N_msm
peak_val, peak_wk = findmax(ovrall_prev_mean)
_wks[peak_wk]
ovrall_prev_mean[peak_wk] |> x -> round(x * 100, sigdigits=3)
@show ovrall_prev_IQR = (ovrall_prev_mean[peak_wk] - prev_cred_int_overall.lb_pred_25[peak_wk] ./ N_msm, prev_cred_int_overall.median_pred[peak_wk] ./ N_msm, ovrall_prev_mean[peak_wk] + prev_cred_int_overall.ub_pred_25[peak_wk] ./ N_msm) .|> x -> round(x * 100, sigdigits=3)
#non-GBMSM peak prev
ovrall_prev_mean_nmsm = prev_cred_int.mean_pred[peak_wk.I[1], 11] ./ (N_uk - N_msm)
ovrall_prev_IQR_nmsm = (ovrall_prev_mean_nmsm - prev_cred_int.lb_pred_25[17, 11] ./ (N_uk - N_msm),
    prev_cred_int.median_pred[peak_wk.I[1], 11] ./ (N_uk - N_msm), ovrall_prev_mean_nmsm + prev_cred_int.ub_pred_25[17, 11] ./ (N_uk - N_msm)) .|> x -> round(x * 100, sigdigits=3)

#Peak prevalence of most sexually active group
prev_mean_mst_active = prev_cred_int.mean_pred[:, 10] ./ N_msm_grp[10]
peak_val, peak_wk = findmax(prev_mean_mst_active)

prev_IQR_mst_active = (prev_mean_mst_active[15] - prev_cred_int.lb_pred_25[15, 10] ./ N_msm_grp[10],
    prev_cred_int.median_pred[15, 10] ./ N_msm_grp[10], prev_mean_mst_active[15] + prev_cred_int.ub_pred_25[15, 10] ./ N_msm_grp[10]) .|> x -> round(x * 100, sigdigits=3)

#Forecast peak week
findmax(cred_int.mean_pred[:, 1])
long_wks[17]
@show cred_int.mean_pred[17, 1], cred_int.mean_pred[17, 1] - cred_int.lb_pred_25[17, 1], cred_int.median_pred[17, 1], cred_int.mean_pred[17, 1] + cred_int.ub_pred_25[17, 1]
findmax(cred_int.mean_pred[:, 2])
long_wks[17]
@show cred_int.mean_pred[17, 2], cred_int.mean_pred[17, 2] - cred_int.lb_pred_25[17, 2], cred_int.median_pred[17, 2], cred_int.mean_pred[17, 2] + cred_int.ub_pred_25[17, 2]

wk = 19
spread = cred_int.mean_pred[wk, 1], cred_int.mean_pred[wk, 1] - cred_int.lb_pred_25[wk, 1], cred_int.median_pred[wk, 1], cred_int.mean_pred[wk, 1] + cred_int.ub_pred_25[wk, 1]
@show spread
wk = 23
spread = cred_int.mean_pred[wk, 1], cred_int.mean_pred[wk, 1] - cred_int.lb_pred_25[wk, 1], cred_int.median_pred[wk, 1], cred_int.mean_pred[wk, 1] + cred_int.ub_pred_25[wk, 1]
@show spread

wk = 19
spread = cred_int.mean_pred[wk, 2], cred_int.mean_pred[wk, 2] - cred_int.lb_pred_25[wk, 2], cred_int.median_pred[wk, 2], cred_int.mean_pred[wk, 2] + cred_int.ub_pred_25[wk, 2]
@show spread
wk = 23
spread = cred_int.mean_pred[wk, 2], cred_int.mean_pred[wk, 2] - cred_int.lb_pred_25[wk, 2], cred_int.median_pred[wk, 2], cred_int.mean_pred[wk, 2] + cred_int.ub_pred_25[wk, 2]
@show spread

#Cumulative forecasts
creds = deepcopy(cred_int_cum_incidence)
mean_total_msm = total_cases[end, 1] + creds.mean_pred[end, 1]
spread = (mean_total_msm, mean_total_msm - creds.lb_pred_25[end, 1], total_cases[end, 1] + creds.median_pred[end, 1], mean_total_msm + creds.ub_pred_25[end, 1]) .|> x -> round(x, digits=1)
@show spread


mean_total_nmsm = total_cases[end, 2] + creds.mean_pred[end, 2]
spread = (mean_total_nmsm, mean_total_nmsm - creds.lb_pred_25[end, 2], total_cases[end, 2] + creds.median_pred[end, 2], mean_total_nmsm + creds.ub_pred_25[end, 2]) .|> x -> round(x, digits=1)
@show spread

#Cumulative no further reduction in transmission potential
creds = deepcopy(cred_int_cum_noredtrans)
mean_total_msm = total_cases[end, 1] + creds.mean_pred[end, 1]
spread = (mean_total_msm, mean_total_msm - creds.lb_pred_25[end, 1], total_cases[end, 1] + creds.median_pred[end, 1], mean_total_msm + creds.ub_pred_25[end, 1]) .|> x -> round(x, digits=1)
@show spread

mean_total_nmsm = total_cases[end, 2] + creds.mean_pred[end, 2]
spread = (mean_total_nmsm, mean_total_nmsm - creds.lb_pred_25[end, 2], total_cases[end, 2] + creds.median_pred[end, 2], mean_total_nmsm + creds.ub_pred_25[end, 2]) .|> x -> round(x, digits=1)
@show spread

#Cumulative no vaccines
creds = deepcopy(cred_int_cum_no_vaccines)
mean_total_msm = total_cases[end, 1] + creds.mean_pred[end, 1]
spread = (mean_total_msm, mean_total_msm - creds.lb_pred_25[end, 1], total_cases[end, 1] + creds.median_pred[end, 1], mean_total_msm + creds.ub_pred_25[end, 1]) .|> x -> round(x, digits=1)
@show spread

mean_total_nmsm = total_cases[end, 2] + creds.mean_pred[end, 2]
spread = (mean_total_nmsm, mean_total_nmsm - creds.lb_pred_25[end, 2], total_cases[end, 2] + creds.median_pred[end, 2], mean_total_nmsm + creds.ub_pred_25[end, 2]) .|> x -> round(x, digits=1)
@show spread

#Cumulative rwc scenario
creds = deepcopy(cred_int_cum_incidence_no_intervention)
mean_total_msm = total_cases[end, 1] + creds.mean_pred[end, 1]
spread = (mean_total_msm, mean_total_msm - creds.lb_pred_25[end, 1], total_cases[end, 1] + creds.median_pred[end, 1], mean_total_msm + creds.ub_pred_25[end, 1]) .|> x -> round(x, digits=1)
@show spread

mean_total_nmsm = total_cases[end, 2] + creds.mean_pred[end, 2]
spread = (mean_total_nmsm, mean_total_nmsm - creds.lb_pred_25[end, 2], total_cases[end, 2] + creds.median_pred[end, 2], mean_total_nmsm + creds.ub_pred_25[end, 2]) .|> x -> round(x, digits=1)
@show spread