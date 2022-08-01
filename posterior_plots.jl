using Distributions, StatsBase, StatsPlots
using LinearAlgebra, RecursiveArrayTools
using OrdinaryDiffEq, ApproxBayes
using JLD2, MLUtils, MCMCChains

## Grab UK data and setup model
include("mpxv_datawrangling.jl");
include("setup_model.jl");

##Load posterior draws and structure
smc = load("smc_posterior_draws_vs3_nrw.jld2")["smc_cng_pnt2"]
param_draws = [part.params for part in smc.particles]

##Create transformations to more interpetable parameters
param_names = [:clique_dispersion, :prob_detect, :mean_inf_period, :prob_transmission,
    :R0_other, :detect_dispersion, :init_infs, :chg_pnt, :sex_trans_red, :other_trans_red]

transformations = [fill(x -> x, 2)
    x -> 1 + mean(Geometric(1 / (1 + x))) # Translate the infectious period parameter into mean infectious period
    fill(x -> x, 2)
    x -> 1 / (x + 1) #Translate "effective sample size" for Beta-Binomial on sampling to overdispersion parameter
    fill(x -> x, 4)]
function col_transformations(X, f_vect)
    for j = 1:size(X, 2)
        X[:, j] = f_vect[j].(X[:, j])
    end
    return X
end

val_mat = smc.parameters |> X -> col_transformations(X, transformations) |> X -> [X[i,j] for i = 1:size(X,1),j = 1:size(X,2), k = 1:1  ]
prior_tuple = smc.setup.prior.distribution
chn = Chains(val_mat, param_names)

##
sample_inf_periods = [1 + rand(Geometric(1 / (1 + μ))) for μ in [θ[3] for θ in param_draws]]
μs = [1 + mean(Geometric(1 / (1 + μ))) for μ in [θ[3] for θ in param_draws]]

num = [sum(sample_inf_periods .== n) for n = 1:maximum(sample_inf_periods)]
bar(num ./ 2000,
    xlims=(0, 30))
## Transform to effective number of groups
post_num_groups = map(α -> sum(rand(DirichletMultinomial(N_msm, α * ones(n_cliques))) .> 0), [θ[1] for θ in param_draws])
histogram(post_num_groups)
# histogram([θ[1] for θ in param_draws],norm = :pdf,fillalpha = 0.3)
# histogram!(rand(prior_tuple[1],2000),norm = :pdf,fillalpha = 0.3)
##
plt_post = plot(chn,
    left_margin=5mm)

# for i = 1:10
#     plot!(plt_post[2*i], prior_tuple[i])
# end
display(plt_post)
savefig(plt_post, "plots/post_plots.png")

##
crn_plt = corner(chn,
    size=(1500, 1500),
    left_margin=5mm, right_margin=5mm)
savefig(crn_plt, "plots/post_crnplot.png")

##

