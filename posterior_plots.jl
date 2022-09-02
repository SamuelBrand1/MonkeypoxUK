using Distributions, StatsBase, StatsPlots
using LinearAlgebra, RecursiveArrayTools
using OrdinaryDiffEq, ApproxBayes
using JLD2, MCMCChains

## Grab UK data and setup model
include("mpxv_datawrangling.jl");
include("setup_model.jl");

##Load posterior draws and structure

smc = MonkeypoxUK.load_smc("posteriors/smc_posterior_draws_2022-08-29.jld2")
param_draws = [part.params for part in smc.particles]

##Create transformations to more interpetable parameters
param_names = [:clique_dispersion, :prob_detect, :mean_inf_period, :prob_transmission,
    :R0_other, :detect_dispersion, :init_infs, :chg_pnt, :sex_trans_red, :other_trans_red,:sex_trans_red_post_WHO, :other_trans_red_post_WHO]

transformations = [fill(x -> x, 2)
    x -> 1 + mean(Geometric(1 / (1 + x))) # Translate the infectious period parameter into mean infectious period
    fill(x -> x, 2)
    x -> 1 / (x + 1) #Translate "effective sample size" for Beta-Binomial on sampling to overdispersion parameter
    fill(x -> x, 4);
    fill(x -> x, 2)]
function col_transformations(X, f_vect)
    for j = 1:size(X, 2)
        X[:, j] = f_vect[j].(X[:, j])
    end
    return X
end

val_mat = smc.parameters |> X -> col_transformations(X, transformations) |> X -> hcat(X[:,1:10],X[:,11].*X[:,4],X[:,12].*X[:,5])  |> X -> [X[i, j] for i = 1:size(X, 1), j = 1:size(X, 2), k = 1:1]
chn = Chains(val_mat, param_names)

write("posteriors/posterior_chain_" * string(wks[end]) * ".jls", chn)

##
prior_tuple = smc.setup.prior.distribution
prior_val_mat = Matrix{Float64}(undef, 10_000, length(prior_tuple))
for j = 1:length(prior_tuple)
    prior_val_mat[:, j] .= rand(prior_tuple[j], 10_000)
end
prior_val_mat = col_transformations(prior_val_mat, transformations)
prior_val_mat[:,11] .= prior_val_mat[:,11].*prior_val_mat[:,4]
prior_val_mat[:,12] .= prior_val_mat[:,11].*prior_val_mat[:,5]
##
pretty_parameter_names = ["Clique size dispersion",
    "Prob. of detection",
    "Mean dur. infectious",
    "Prob. trans. per sexual contact",
    "Non-sexual R0",
    "Prob. of detect. dispersion",
    "Init. Infs scale",
    "Timing: 1st change point",
    "Sex. trans. reduction: 1st cng pnt",
    "Other trans. reduction: 1st cng pnt",
    "Sex. trans. reduction: WHO cng pnt",
    "Other. trans. reduction: WHO cng pnt"]

post_plt = plot(; layout=(6, 2),
    size=(800, 2000), dpi=250,
    left_margin=10mm,
    right_margin=10mm)

for j = 1:length(prior_tuple)
    histogram!(post_plt[j], val_mat[:, j],
        norm=:pdf,
        fillalpha=0.5,
        nbins=100,
        lw=0.5,
        alpha=0.1,
        lab="",
        color=1,
        title=string(pretty_parameter_names[j]))
    histogram!(post_plt[j], prior_val_mat[:, j],
        norm=:pdf,
        fillalpha=0.5,
        alpha=0.1,
        color=2,
        nbins=100,
        lab="")
    density!(post_plt[j], val_mat[:, j],
        lw=3,
        color=1,
        lab="Posterior")
    density!(post_plt[j], prior_val_mat[:, j],
        lw=3,
        color=2,
        lab="Prior")
end
display(post_plt)
savefig(post_plt, "posteriors/post_plot" * string(wks[end])  * ".png")

##
crn_plt = corner(chn,
    size=(1500, 1500),
    left_margin=5mm, right_margin=5mm)
savefig(crn_plt, "posteriors/post_crnplot" * string(wks[end]) * ".png")

##

@show chn