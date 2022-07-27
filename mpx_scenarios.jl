using Distributions, StatsBase, StatsPlots
using LinearAlgebra, RecursiveArrayTools
using OrdinaryDiffEq, ApproxBayes
using JLD2, MLUtils

## Grab UK data and model set up

include("mpxv_datawrangling.jl");
include("setup_model.jl");

##Load posterior draws
filenames = ["draws1_nrw.jld2","draws2_nrw.jld2"]
smcs = [load(filename)["smc_cng_pnt"] for filename in filenames];
particles = smcs[1].particles
particles = [particles;smcs[2].particles]
param_draws = [part.params for part in particles]

##Generate predictions with no change

long_wks = [wks;[wks[end] + Day(7*k) for k = 1:12]]
long_mpxv_wkly = [mpxv_wkly;zeros(12)]

preds = map(θ -> mpx_sim_function_chp(θ,constants,long_mpxv_wkly)[2],param_draws)

##Simulation projections

median_pred = [median([preds[n][wk] for n = 1:length(preds)])   for wk in 1:length(long_wks)]
lb_pred_25 = median_pred .- [quantile([preds[n][wk] for n = 1:length(preds)],0.25)   for wk in 1:length(long_wks)]
lb_pred_025 = median_pred .- [quantile([preds[n][wk] for n = 1:length(preds)],0.025)   for wk in 1:length(long_wks)]
ub_pred_25 = [quantile([preds[n][wk] for n = 1:length(preds)],0.75)   for wk in 1:length(long_wks)] .- median_pred
ub_pred_025 = [quantile([preds[n][wk] for n = 1:length(preds)],0.975)   for wk in 1:length(long_wks)] .- median_pred

##
plt = plot(;ylabel = "Weekly cases",
            title = "UK MPVX reasonable worst case",yscale = :log10,
            legend = :topleft,
            yticks = ([1,10,100,1000],[1,10,100,1000]))
plot!(plt,long_wks,median_pred,ribbon = (lb_pred_025,ub_pred_025),lw = 0,
        color = :purple, fillalpha = 0.3, lab = "")
plot!(plt,long_wks,median_pred,ribbon = (lb_pred_25,ub_pred_25),lw = 3,
        color = :purple, fillalpha = 0.3,lab = "Projection")        
scatter!(plt,wks[[1,2,end]],mpxv_wkly[[1,2,end]],lab = "Data (not used in fitting)",ms = 6,color = :green)
scatter!(plt,wks[3:(end-1)],mpxv_wkly[3:(end-1)],lab = "Data (used in fitting)",ms = 6,color = :blue)

display(plt)


## Public health emergency effect forecast


