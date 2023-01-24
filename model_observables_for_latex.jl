using Base: emphasize
# This script outputs LaTeX script which is used in the main document by defining a set of commands.

using Distributions, StatsBase, StatsPlots, Plots.PlotMeasures
using LinearAlgebra, RecursiveArrayTools
using OrdinaryDiffEq, ApproxBayes, CSV, DataFrames, Dates
using JLD2, MCMCChains
using MonkeypoxUK
using LaTeXStrings, Latexify

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


##Load posterior draws and structure


# Main model

param_names = [
    :metapop_size_dispersion,
    :prob_detect,
    :prob_transmission,
    :R0_other,
    :detect_dispersion,
    :init_infs,
    :chg_pnt,
    :sex_trans_red,
    :other_trans_red,
    :sex_trans_red_post_WHO,
    :other_trans_red_post_WHO,
]

transformations = [
    fill(x -> x, 4)
    x -> 1 / (x + 1) #Translate "effective sample size" for Beta-Binomial on sampling to overdispersion parameter
    fill(x -> x, 4)
    fill(x -> x, 2)
]

function col_transformations(X, f_vect)
    for j = 1:size(X, 2)
        X[:, j] = f_vect[j].(X[:, j])
    end
    return X
end

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

param_idxs_no_ngbmsm_chg = [trues(7);true;false;true;false]

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

param_idxs_one_metapop = [false;trues(6);true;false;true;false]

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

param_idxs = trues(11)

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

param_idxs_no_bv_cng = [trues(7);falses(4)]

model_str_to_prior = Dict("no_ngbmsm_chg" => (prior_vect_no_ngbmsm_chg, param_idxs_no_ngbmsm_chg),
                            "no_bv_cng" => (prior_vect_no_bv_cng, param_idxs_no_bv_cng),
                            "one_metapop" => (prior_vect_one_metapop, param_idxs_one_metapop),
                            "" => (prior_vect, param_idxs))

##

function get_MCMC_chn(date_str, description_str)
    param_draws = load("posteriors/posterior_param_draws_" * date_str * description_str * ".jld2")["param_draws"]
    idxs = model_str_to_prior[description_str][2]
    priors = model_str_to_prior[description_str][1][idxs]

    ## Create a Chains object from MCMCChains

    param_mat = [p[j] for p in param_draws, j = findall(idxs)]
    # val_mat = smc.parameters |> X -> col_transformations(X, transformations) |> X -> hcat(X[:,1:10],X[:,11].*X[:,4],X[:,12].*X[:,5])  |> X -> [X[i, j] for i = 1:size(X, 1), j = 1:size(X, 2), k = 1:1]
    # val_mat = smc.parameters |> X -> col_transformations(X, transformations) |> X -> [X[i, j] for i = 1:size(X, 1), j = 1:size(X, 2), k = 1:1]
    val_mat =
        param_mat |>
        X -> col_transformations(X, transformations[idxs]) |>
        X -> [X[i, j] for i = 1:size(X, 1), j = 1:size(X, 2), k = 1:1]
    chn = Chains(val_mat, param_names[idxs])
    return chn, priors
end

## Gather posterior means and quantiles

date_str = "2022-09-26"
description_str = "no_ngbmsm_chg" #<---- This is the main model
# description_str = "no_bv_cng" #<---- This is the version of the model with no behavioural change
# description_str = "one_metapop" #<--- This is the version of the model with no metapopulation structure
# description_str = "" #<--- this is the older version main model

chn, priors = get_MCMC_chn(date_str, description_str)

posterior_means = mean(chn)
quantiles = quantile(chn)

R0_other_idxs = posterior_means.nt.parameters .== :R0_other
prob_trans_idxs = posterior_means.nt.parameters .== :prob_transmission
chg_pnt_idxs = posterior_means.nt.parameters .== :chg_pnt


R0_other_mean = posterior_means.nt.mean[R0_other_idxs][1] |> x -> round(x, sigdigits = 3)
R0_other_q = (quantiles.nt[Symbol("2.5%")][R0_other_idxs][1], quantiles.nt[Symbol("97.5%")][R0_other_idxs][1]) .|> x -> round(x, sigdigits = 3)
prob_trans_mean = posterior_means.nt.mean[prob_trans_idxs][1] * 100 |> x -> round(x, sigdigits = 3) # Put in percentage form
prob_trans_q = (quantiles.nt[Symbol("2.5%")][prob_trans_idxs][1] * 100, quantiles.nt[Symbol("97.5%")][prob_trans_idxs][1] * 100) .|> x -> round(x, sigdigits = 3)

chg_pnt_mean = posterior_means.nt.mean[chg_pnt_idxs][1] |> x -> round(x, sigdigits = 3) |> t -> Date(2021, 12, 31) + Day(Int(t)) |> d -> string(day(d))*"-"*monthname(d)
chg_pnt_q = (quantiles.nt[Symbol("2.5%")][chg_pnt_idxs][1], quantiles.nt[Symbol("97.5%")][chg_pnt_idxs][1]) .|> x -> round(x, sigdigits = 3) .|> t -> Date(2021, 12, 31) + Day(Int(t)) .|> d -> string(day(d))*"-"*monthname(d)


function mean_ci_string(param_name)
    param_idx = posterior_means.nt.parameters .== param_name
    m = posterior_means.nt.mean[param_idx][1] |> x -> round(x, sigdigits = 3)
    qs = (quantiles.nt[Symbol("2.5%")][param_idx][1], quantiles.nt[Symbol("97.5%")][param_idx][1]) .|> x -> round(x, sigdigits = 3)
    return "$(m) ($(qs[1]), $(qs[2]))"
end


## Parameter output to LaTeX strings

output = ""
output = output * raw"\newcommand{\Rotherfit}{"
output = output * "$(R0_other_mean) ($(R0_other_q[1]) -- $(R0_other_q[2])}\n"
output = output * raw"\newcommand{\probtransmissionfit}{"
output = output * LaTeXString("$(prob_trans_mean)\\% ($(prob_trans_q[1]) -- $(prob_trans_q[2])\\%}\n")


## Posterior distribution table

posterior_distrib_str = raw"""
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}clll@{\extracolsep{\fill}}}
\hline
Parameter & \multicolumn{1}{c}{\begin{tabular}[c]{@{}c@{}}Posterior mean\\ (95\% CI)\end{tabular}} & \multicolumn{1}{c}{Prior distribution} & \multicolumn{1}{c}{Description} \\ 
\hline
\multicolumn{4}{c}{Population structure and initial condition parameters} \\ \hline
$\alpha_m$ & """ *
mean_ci_string(:metapop_size_dispersion)  *
raw"""
& $Exp(1)$ & \begin{tabular}[c]{@{}l@{}}Dispersion parameter for\\ metapopulation sizes.\end{tabular}  \\ 
\hline
$\iota_0$  & """ *
mean_ci_string(:init_infs) * 
raw"""
& $LogNormal(\ln(5), 1)$  & \begin{tabular}[c]{@{}l@{}}Scale parameter for\\ the number of initial\\ infected people.\end{tabular}\\ 
\hline
\multicolumn{4}{c}{Detection probability parameters} \\ \hline
$p_d$  & """ * 
mean_ci_string(:prob_detect) * 
raw"""
& $Beta(5,5)$ & \begin{tabular}[c]{@{}l@{}}Mean weekly probability\\ of case detection.\\ Weekly probability of\\ detection was random\\ with $P_w \sim Beta(\alpha,\beta)$\\ with $\mathbb{E}[P_w]=p_d$.\end{tabular}  \\ \hline
$\Phi_d=\frac{1}{M+1}$ & """ * 
mean_ci_string(:detect_dispersion) * 
raw"""
& $M \sim Gamma(3,1000/3)$  & \begin{tabular}[c]{@{}l@{}}Dispersion of weekly\\ probability of case\\ detection (M is the\\ effective sample size\\ $\alpha + \beta$ for the weekly\\ $P_w \sim Beta(\alpha,\beta)$\\ distributed probability\\ of case detection).\end{tabular} \\ \hline
\multicolumn{4}{c}{Baseline transmission parameters}  \\ \hline
$p_{gbmsm}(0)$ & """ * 
mean_ci_string(:prob_transmission) * 
raw"""& $Uniform(0,1)$  & \begin{tabular}[c]{@{}l@{}}Baseline probability\\ of transmission per\\ sexual contact.\end{tabular}\\ \hline
$R_{other}(0)$  & """ * 
mean_ci_string(:R0_other) * 
raw"""
& $LogNormal(\ln(0.25),1)$ & \begin{tabular}[c]{@{}l@{}}Baseline reproductive\\ number non-GBMSM\\ sexual contacts.\end{tabular} \\ \hline
\multicolumn{4}{c}{Behaviour and risk change point parameters}  \\ \hline
$T_1$ & \begin{tabular}[c]{@{}l@{}}""" * chg_pnt_mean * raw"\\(" * chg_pnt_q[1] * raw", " * chg_pnt_q[2] * 
raw""")\end{tabular}  & $U$(15 May, 18 July) & \begin{tabular}[c]{@{}l@{}}Change point time for\\ reduction in transmission\\ due to awareness of MPX.\end{tabular} \\ \hline
$\rho_{gbmsm,1}$  & """ *
mean_ci_string(:sex_trans_red) * 
raw"""& $Beta(1.5,1.5)$ & \begin{tabular}[c]{@{}l@{}}Reduction in GBMSM\\ reproduction number after\\ change point at $T_1$.\end{tabular} \\ \hline
$\rho_{gbmsm,2}$ & """ *
mean_ci_string(:sex_trans_red_post_WHO) * 
raw""" & \begin{tabular}[c]{@{}l@{}}$\rho_{gbmsm,2} \sim Beta(1.5,1.5)$\end{tabular} & \begin{tabular}[c]{@{}l@{}}Further reduction in GBMSM\\ reproduction number after\\ WHO announcement of\\ PHEIC.\end{tabular}  \\ \hline
\end{tabular*}"""

output = output * raw"\newcommand{\posteriortable}{"
output = output * posterior_distrib_str * "}\n"



## Reproductive ratio for within GBMSM and overall
eff_infectious_period = (1 / p_inf) + (epsilon / p_incubation)
p_transes = chn[:prob_transmission][:] |> Vector
red1 = chn[:sex_trans_red][:] |> Vector 
red2 = chn[:sex_trans_red_post_WHO][:] |> Vector 


R_gbmsms = eff_infectious_period .* mean(mean_daily_cnts) .* p_transes
R_gbmsms_red1 = eff_infectious_period .* mean(mean_daily_cnts) .* p_transes .* (1 .- red1)
R_gbmsms_red2 = eff_infectious_period .* mean(mean_daily_cnts) .* p_transes .* (1 .- red1) .* (1 .- red2)

mean_R_gbmsm = mean(R_gbmsms) |> x -> round(x, sigdigits = 3)
qs_R_gbmsm = quantile(R_gbmsms, [0.025, 0.975]) .|> x -> round(x, sigdigits = 3)
mean_R_gbmsm_red1 = mean(R_gbmsms_red1) |> x -> round(x, sigdigits = 3)
qs_R_gbmsm_red1 = quantile(R_gbmsms_red1, [0.025, 0.975]) .|> x -> round(x, sigdigits = 3)
mean_R_gbmsm_red2 = mean(R_gbmsms_red2) |> x -> round(x, sigdigits = 3)
qs_R_gbmsm_red2 = quantile(R_gbmsms_red2, [0.025, 0.975]) .|> x -> round(x, sigdigits = 3)
mean_red_1 = mean(red1) |> x -> round(x, sigdigits = 3) * 100
qs_red_1 = quantile(red1, [0.025, 0.975]) .|> x -> round(x, sigdigits = 3) * 100

output = output * raw"\newcommand{\gbmsmR}{" * "$(mean_R_gbmsm) ($(qs_R_gbmsm[1]) -- $(qs_R_gbmsm[2])}\n"
output = output * raw"\newcommand{\efftransmissionperiod}{" * "$(round(eff_infectious_period, sigdigits = 3))}\n"
output = output * raw"\newcommand{\gbmsmRredbv}{" * "$(mean_R_gbmsm_red1) ($(qs_R_gbmsm_red1[1]) -- $(qs_R_gbmsm_red1[2])}\n"
output = output * raw"\newcommand{\gbmsmRredwho}{" * "$(mean_R_gbmsm_red2) ($(qs_R_gbmsm_red2[1]) -- $(qs_R_gbmsm_red2[2])}\n"
output = output * raw"\newcommand{\reductionone}{" * string(mean_red_1) * raw"\%}\n"

##Calculate orignal R₀ and latest R(t)
"""
function construct_next_gen_mat(params, constants, susceptible_prop, vac_rates; vac_eff::Union{Nothing,Number} = nothing)

Construct the next generation matrix `G` in orientation: 
> `G_ij = E[# Infected people in group j due to one in group i]`
Returns `(Real(eigvals(G)[end]), G)`. NB: `Real(eigvals(G)[end])` is the leading eignvalue of `G` i.e. the reproductive number.
"""
function construct_next_gen_mat(
    params,
    constants,
    susceptible_prop,
    vac_rates;
    vac_eff::Union{Nothing,Number} = nothing,
)
    #Get parameters 
    α_choose,
    p_detect,
    p_trans,
    R0_other,
    M,
    init_scale,
    chp_t,
    trans_red,
    trans_red_other,
    trans_red2,
    trans_red_other2 = params

    #Get constant data
    N_total,
    N_msm,
    ps,
    ms,
    ingroup,
    ts,
    α_incubation,
    γ_eff,
    epsilon,
    n_cliques,
    wkly_vaccinations,
    vac_effectiveness,
    chp_t2,
    weeks_to_change = constants


    if ~isnothing(vac_eff)
        vac_effectiveness = vac_eff
    end
    mean_inf_period = epsilon * (1 / (1 - exp(-α_incubation))) + (1 / (1 - exp(-γ_eff)))

    # R0_other = R0_other + 2.5
    # p_trans = 0.0
    #Calculate next gen matrix G_ij = E[#Infections in group j due to a single infected person in group i]
    _A =
        (ms .* (susceptible_prop .+ (vac_rates .* (1.0 .- vac_effectiveness)))') .*
        (mean_inf_period .* p_trans ./ length(ms))  #Sexual transmission within MSM
    A = _A .+ (R0_other / N_uk) .* repeat(ps' .* N_msm, 10) #Other routes of transmission MSM -> MSM
    B = (R0_other * (N_uk - N_msm) / N_total) .* ones(10) # MSM transmission to non MSM
    C = (R0_other / N_uk) .* ps' .* N_msm  #Non-msm transmission to MSM
    D = [(R0_other * (N_uk - N_msm) / N_total)]# Non-MSM transmission to non-MSM
    G = [A B; C D]
    return Real(eigvals(G)[end]), G
end
param_draws = load("posteriors/posterior_param_draws_" * date_str * description_str * ".jld2")["param_draws"]

## Calculate the original R0 with next gen matrix method and lastest R(t)
R0s = map(
    θ -> construct_next_gen_mat(
        θ,
        constants,
        [ones(10); zeros(0)],
        [zeros(10); fill(1.0, 0)],
    )[1],
    param_draws,
)

m = mean(R0s) |> x -> round(x, sigdigits = 3)
qs = quantile(R0s, [0.025, 0.975]) .|> x -> round(x, sigdigits = 3)
output = output * raw"\newcommand{\totalR}{" * "$(m) ($(qs[1]) -- $(qs[2])}\n"

##

open("model_output.tex", "w") do io
    write(io, output)
end;
