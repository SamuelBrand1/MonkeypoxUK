using Distributions, StatsBase, StatsPlots
using LinearAlgebra, RecursiveArrayTools
using OrdinaryDiffEq, ApproxBayes
using JLD2, MCMCChains
using MonkeypoxUK
using ColorSchemes
## Grab UK data and setup model
include("mpxv_datawrangling.jl");
include("setup_model.jl");

## Generate an ensemble of forecasts

seq_wks = [wks[1:5], wks[1:9], wks[1:13], wks]
seq_mpxv_wklys = [mpxv_wkly[1:5, :], mpxv_wkly[1:9, :], mpxv_wkly[1:13, :], mpxv_wkly]

function load_smc(wks)
    wk = wks[end]
    load("posteriors/posterior_param_draws_" * string(wk) * ".jld2")["param_draws"]
end

seq_param_draws = map(load_smc, seq_wks)

seq_forecasts = map((param_draws, wks, mpxv_wkly) -> forecast = generate_forecast_projection(param_draws, wks, mpxv_wkly, constants),
    seq_param_draws,
    seq_wks,
    seq_mpxv_wklys)

##
preds = [[x[1] for x in forecast] for forecast in seq_forecasts]
seq_creds = MonkeypoxUK.cred_intervals.(preds)
long_wks = [wks; [wks[end] + Day(7 * k) for k = 1:12]]
long_mpxv_wkly = [mpxv_wkly; zeros(12, 2)]
##

seq_proj_msm = plot(; ylabel="Weekly cases",
        title="UK Monkeypox Sequential Projections (MSM)",# yscale=:log10,
        legend=:topleft,
        # yticks=([1, 2, 11, 101, 1001], [0, 1, 10, 100, 1000]),
        # ylims=(0.8, 3001),
        xticks=([Date(2022, 5, 1) + Month(k) for k = 0:5], [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:5]),
        left_margin=5mm,
        size=(800, 600), dpi=250,
        tickfont=11, titlefont=18, guidefont=18, legendfont=11)
plot!(seq_proj_msm,long_wks[5:17],seq_creds[1].mean_pred[5:17, 1], color=get(ColorSchemes.cool,5/15),
    ribbon=(seq_creds[1].lb_pred_25[5:17, 1], seq_creds[1].ub_pred_25[5:17, 1]),
    fillalpha = 0.3, legend = :topleft,lab = seq_wks[1][end],lw = 0)
plot!(seq_proj_msm,long_wks[9:21],seq_creds[2].mean_pred[9:21, 1], color=get(ColorSchemes.cool,9/15),
    ribbon=(seq_creds[2].lb_pred_25[9:21, 1], seq_creds[2].ub_pred_25[9:21, 1]),
    fillalpha=0.3, lab=seq_wks[2][end],lw = 0)
plot!(seq_proj_msm,long_wks[13:25],seq_creds[3].mean_pred[13:25, 1], color=get(ColorSchemes.cool,13/15),
    ribbon=(seq_creds[3].lb_pred_25[13:25, 1], seq_creds[3].ub_pred_25[13:25, 1]),
    fillalpha=0.3, lab=seq_wks[3][end],lw = 0)
plot!(seq_proj_msm,long_wks[15:27],seq_creds[4].mean_pred[15:27, 1], color=get(ColorSchemes.cool,15/15),
    ribbon=(seq_creds[4].lb_pred_25[15:27, 1], seq_creds[4].ub_pred_25[15:27, 1]),
    fillalpha=0.3, lab=seq_wks[4][end],lw = 0)    

plot!(seq_proj_msm,long_wks[5:17],seq_creds[1].mean_pred[5:17, 1], color=get(ColorSchemes.cool,5/15),
    lab = "",lw = 3)
plot!(seq_proj_msm,long_wks[9:21],seq_creds[2].mean_pred[9:21, 1], color=get(ColorSchemes.cool,9/15),
    lab="",lw = 3)
plot!(seq_proj_msm,long_wks[13:25],seq_creds[3].mean_pred[13:25, 1], color=get(ColorSchemes.cool,13/15),
    lab="",lw = 3)
plot!(seq_proj_msm,long_wks[15:27],seq_creds[4].mean_pred[15:27, 1], color=get(ColorSchemes.cool,15/15),
    lab="",lw = 3)    
# plot!(15:27,seq_creds[4].mean_pred[15:27, 1], color=get(ColorSchemes.cool,15/15),
#     lab="",lw = 3)    
# scatter!(mpxv_wkly[1:end-1,1],color = :black,lab= "Data")
scatter!(seq_proj_msm,wks[1:(end-1)],mpxv_wkly[1:(end-1), 1],
        lab="Data",
        ms=6, color=:black)

##

seq_proj_nmsm = plot(; ylabel="Weekly cases",
        title="UK Monkeypox Sequential Projections (non-MSM)",# yscale=:log10,
        legend=:topleft,
        # yticks=([1, 2, 11, 101, 1001], [0, 1, 10, 100, 1000]),
        # ylims=(0.8, 3001),
        xticks=([Date(2022, 5, 1) + Month(k) for k = 0:5], [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:5]),
        left_margin=5mm,
        size=(800, 600), dpi=250,
        tickfont=11, titlefont=18, guidefont=18, legendfont=11)
plot!(seq_proj_nmsm,long_wks[5:17],seq_creds[1].mean_pred[5:17, 2], color=get(ColorSchemes.cool,5/15),
    ribbon=(seq_creds[1].lb_pred_25[5:17, 2], seq_creds[1].ub_pred_25[5:17, 2]),
    fillalpha = 0.3, legend = :topleft,lab = seq_wks[1][end],lw = 0)
plot!(seq_proj_nmsm,long_wks[9:21],seq_creds[2].mean_pred[9:21, 2], color=get(ColorSchemes.cool,9/15),
    ribbon=(seq_creds[2].lb_pred_25[9:21, 2], seq_creds[2].ub_pred_25[9:21, 2]),
    fillalpha=0.3, lab=seq_wks[2][end],lw = 0)
plot!(seq_proj_nmsm,long_wks[13:25],seq_creds[3].mean_pred[13:25, 2], color=get(ColorSchemes.cool,13/15),
    ribbon=(seq_creds[3].lb_pred_25[13:25, 2], seq_creds[3].ub_pred_25[13:25, 2]),
    fillalpha=0.3, lab=seq_wks[3][end],lw = 0)
plot!(seq_proj_nmsm,long_wks[15:27],seq_creds[4].mean_pred[15:27, 2], color=get(ColorSchemes.cool,15/15),
    ribbon=(seq_creds[4].lb_pred_25[15:27, 2], seq_creds[4].ub_pred_25[15:27, 2]),
    fillalpha=0.3, lab=seq_wks[4][end],lw = 0)    

plot!(seq_proj_nmsm,long_wks[5:17],seq_creds[1].mean_pred[5:17, 2], color=get(ColorSchemes.cool,5/15),
    lab = "",lw = 3)
plot!(seq_proj_nmsm,long_wks[9:21],seq_creds[2].mean_pred[9:21, 2], color=get(ColorSchemes.cool,9/15),
    lab="",lw = 3)
plot!(seq_proj_nmsm,long_wks[13:25],seq_creds[3].mean_pred[13:25, 2], color=get(ColorSchemes.cool,13/15),
    lab="",lw = 3)
plot!(seq_proj_nmsm,long_wks[15:27],seq_creds[4].mean_pred[15:27, 2], color=get(ColorSchemes.cool,15/15),
    lab="",lw = 3)    
# plot!(15:27,seq_creds[4].mean_pred[15:27, 1], color=get(ColorSchemes.cool,15/15),
#     lab="",lw = 3)    
# scatter!(mpxv_wkly[1:end-1,1],color = :black,lab= "Data")
scatter!(seq_proj_nmsm,wks[1:(end-1)],mpxv_wkly[1:(end-1), 2],
        lab="Data",
        ms=6, color=:black)        

##
        