using Distributions, StatsBase, StatsPlots
using LinearAlgebra, RecursiveArrayTools
using OrdinaryDiffEq, ApproxBayes
using JLD2, MLUtils,MCMCChains

## Grab UK data and setup model
include("mpxv_datawrangling.jl");
include("setup_model.jl");

##Load posterior draws and structure
smc = load("smc_posterior_draws.jld2")["smc_cng_pnt"]
param_draws = [part.params for part in smc.particles]



##Define Prior

prior_tuple = smc.setup.prior.distribution
nparam = smc.setup.nparams
val_mat = [smc.parameters[i,j] for i = 1:size(smc.parameters,1), j = 1:size(smc.parameters,2), k = 1:1]
param_names = [:clique_dispersion,:prob_detect,:mean_inf_period,:prob_transmission,
                :R0_other,:detect_dispersion,:init_infs,:chg_pnt,:sex_trans_red,:other_trans_red] 

chn = Chains(val_mat,param_names)
## Transform to effective number of groups
post_num_groups = map(α -> sum(rand(DirichletMultinomial(N_msm,α * ones(n_cliques))) .> 0),[θ[1] for θ in param_draws])
histogram(post_num_groups)
# histogram([θ[1] for θ in param_draws],norm = :pdf,fillalpha = 0.3)
# histogram!(rand(prior_tuple[1],2000),norm = :pdf,fillalpha = 0.3)
##
plt_post = plot(chn,
            left_margin = 5mm)

for i = 1:10
    plot!(plt_post[2*i],prior_tuple[i])
end
display(plt_post)
savefig(plt_post,"plots/post_plots.png")

corner(chn,
        size = (1500,1500),
        left_margin = 5mm,right_margin = 5mm)

##

