using Distributions, StatsBase, StatsPlots, Plots.PlotMeasures
using LinearAlgebra, RecursiveArrayTools
using OrdinaryDiffEq, ApproxBayes, CSV, DataFrames, Dates
using JLD2, MCMCChains
using MonkeypoxUK

## Grab UK data and setup model
past_mpxv_data_inferred =
    CSV.File("data/weekly_data_imputation_2022-09-30.csv", missingstring = "NA") |>
    DataFrame

colname = "seqn_fit5"
inferred_prop_na_msm = past_mpxv_data_inferred[:, colname] |> x -> x[.~ismissing.(x)]
mpxv_wkly =
    past_mpxv_data_inferred[1:size(inferred_prop_na_msm, 1), ["gbmsm", "nongbmsm"]] .+
    past_mpxv_data_inferred[1:size(inferred_prop_na_msm, 1), "na_gbmsm"] .*
    hcat(inferred_prop_na_msm, 1.0 .- inferred_prop_na_msm) |> Matrix

wks = Date.(past_mpxv_data_inferred.week[1:size(mpxv_wkly, 1)], DateFormat("dd/mm/yyyy"))

# Leave out first two weeks because reporting changed in early May
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

##Load posterior draws and structure

date_str = "2022-09-26"

description_str = "no_ngbmsm_chg" #<---- This is the main model
# description_str = "no_bv_cng" #<---- This is the version of the model with no behavioural change
# description_str = "one_metapop" #<--- This is the version of the model with no metapopulation structure
# description_str = "" #<--- this is the older version main model

param_draws = load("posteriors/posterior_param_draws_" * date_str * description_str * ".jld2")["param_draws"]

## Create size distribution plot for the meta population sizes
n_metapop = 50
α_metapop_draws = [θ[1] for θ in param_draws]
size_distribution =
    α_metapop_draws .|>
    α -> rand(DirichletMultinomial(N_msm, α * ones(n_cliques))) |> x -> sort(x, rev = true)
size_distribution_mat = [
    size_distribution[i][j] for i = 1:length(size_distribution),
    j = 1:length(size_distribution[1])
]
mean_sizes = mean(size_distribution_mat, dims = 1)[:] #mean(size_distribution_mat,dims = 1)[:]
lb =
    mean_sizes .- [
        quantile(size_distribution_mat[:, metapop], 0.025) for
        metapop = 1:size(size_distribution_mat, 2)
    ]
ub =
    [
        quantile(size_distribution_mat[:, metapop], 0.975) for
        metapop = 1:size(size_distribution_mat, 2)
    ] .- mean_sizes

plt_grp_size = bar(
    mean_sizes ./ N_msm,
    yerrors = (lb, ub) ./ N_msm,
    lab = "Posterior mean group size",
    title = "Ordered metapopulation clique sizes",
    xlabel = "Clique size rank",
    ylabel = "Proportion of GBMSM in clique",
    xticks = [1; 5:5:50],
    size = (800, 600),
    dpi = 250,
    left_margin = 5mm,
    guidefont = 16,
    tickfont = 13,
    titlefont = 24,
    legendfont = 16,
    right_margin = 5mm,
)
display(plt_grp_size)
savefig(plt_grp_size, "plots/metapopulation_sizes.png")
##Create transformations to more interpetable parameters
all_priors, idxs = model_str_to_prior[description_str]
priors = all_priors[idxs]
param_mat = [p[j] for p in param_draws, j = findall(idxs)]
names = param_names[idxs]
val_mat =
    param_mat |>
    X ->
        col_transformations(X, transformations) |>
        X -> [X[i, j] for i = 1:size(X, 1), j = 1:size(X, 2), k = 1:1]
chn = Chains(val_mat, names)

CSV.write("posteriors/posterior_chain_" * date_str * description_str * ".csv", DataFrame(chn))

##
pretty_parameter_names = [
    "Metapop. size dispersion",
    "Prob. of case detection",
    "Prob. trans. per sexual contact",
    "Other R0",
    "Prob. of detect. dispersion",
    "Init. Infs scale",
    "Timing: 1st change point",
    "Sex. trans. reduction: 1st cng pnt",
    "Other trans. reduction: 1st cng pnt",
    "Sex. trans. reduction: WHO cng pnt",
    "Other. trans. reduction: WHO cng pnt",
]
pretty_names = pretty_parameter_names[idxs]

detection_dispersion_prior_draws = rand(priors[5], 10_000) .|> x -> 1 / (x + 1)

post_plt = plot(;
    layout = (3, 3),
    size = (1500, 1500),
    dpi = 250,
    left_margin = 10mm,
    right_margin = 10mm,
)

for (j, prior) in enumerate(priors)
    histogram!(
        post_plt[j],
        val_mat[:, j, 1][:],
        norm = :pdf,
        fillalpha = 0.3,
        nbins = 100,
        lw = 0.5,
        alpha = 0.1,
        lab = "",
        color = 1,
        title = string(pretty_names[j]),
        titlefont = 18,
        legendfont = 14,
    )
    density!(post_plt[j], val_mat[:, j], lw = 3, color = 1, lab = "Posterior")
    if j != 5
        plot!(post_plt[j], prior, lw = 3, color = 2, lab = "Prior")
    else        
        density!(post_plt[j], detection_dispersion_prior_draws, lw = 3, color = 2, lab = "Prior")
    end
    
end

##
plot!(post_plt[4], xlims = (0,1.5))
plot!(post_plt[5], xlims = (0,0.025))
plot!(post_plt[6], xlims = (0,50))
display(post_plt)


##
savefig(post_plt, "plots/post_plot" * date_str * description_str * ".png")

##
crn_plt = corner(chn, size = (2000, 2000), left_margin = 5mm, right_margin = 5mm)
savefig(crn_plt, "plots/post_crnplot" * date_str * description_str  * ".pdf")

##
