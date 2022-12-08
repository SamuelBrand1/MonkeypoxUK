using Distributions, StatsBase, StatsPlots, Plots.PlotMeasures
using LinearAlgebra, RecursiveArrayTools
using OrdinaryDiffEq, ApproxBayes, CSV, DataFrames, Dates
using JLD2, MCMCChains
using MonkeypoxUK

## Grab UK data and setup model
past_mpxv_data_inferred = CSV.File("data/weekly_data_imputation_2022-09-30.csv",
                                missingstring = "NA") |> DataFrame

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

## Comment out to use latest data rather than reterospective data

colname = "seqn_fit5"
inferred_prop_na_msm = past_mpxv_data_inferred[:, colname] |> x -> x[.~ismissing.(x)]
mpxv_wkly =
    Matrix(
        past_mpxv_data_inferred[1:size(inferred_prop_na_msm, 1), ["gbmsm", "nongbmsm"]],
    ) .+
    Vector(past_mpxv_data_inferred[1:size(inferred_prop_na_msm, 1), "na_gbmsm"]) .*
    hcat(inferred_prop_na_msm, 1.0 .- inferred_prop_na_msm)
wks = Date.(past_mpxv_data_inferred.week[1:size(mpxv_wkly, 1)], DateFormat("dd/mm/yyyy"))


##Load posterior draws and structure

smc = MonkeypoxUK.load_smc("posteriors/smc_posterior_draws_2022-09-26.jld2")
param_draws = load("posteriors/posterior_param_draws_2022-09-26.jld2")["param_draws"]

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
savefig(plt_grp_size, "plots/clique_size.png")
##Create transformations to more interpetable parameters
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

param_mat = [p[j] for p in param_draws, j = 1:length(param_names)]
# val_mat = smc.parameters |> X -> col_transformations(X, transformations) |> X -> hcat(X[:,1:10],X[:,11].*X[:,4],X[:,12].*X[:,5])  |> X -> [X[i, j] for i = 1:size(X, 1), j = 1:size(X, 2), k = 1:1]
# val_mat = smc.parameters |> X -> col_transformations(X, transformations) |> X -> [X[i, j] for i = 1:size(X, 1), j = 1:size(X, 2), k = 1:1]
val_mat =
    param_mat |>
    X ->
        col_transformations(X, transformations) |>
        X -> [X[i, j] for i = 1:size(X, 1), j = 1:size(X, 2), k = 1:1]
chn = Chains(val_mat, param_names)

CSV.write("posteriors/posterior_chain_" * string(wks[end]) * ".csv", DataFrame(chn))

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
    mean_inf_period,
    p_trans,
    R0_other,
    M,
    init_scale,
    chp_t,
    trans_red,
    trans_red_other,
    scale_trans_red2,
    scale_red_other2 = params
    #Get constant data
    N_total,
    N_msm,
    ps,
    ms,
    ingroup,
    ts,
    α_incubation,
    n_cliques,
    wkly_vaccinations,
    vac_effectiveness,
    chp_t2 = copy(constants)

    if ~isnothing(vac_eff)
        vac_effectiveness = vac_eff
    end

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
@show round(mean(R0s), digits = 2), round.(quantile(R0s, [0.1, 0.9]), digits = 2)

##
prior_tuple = smc.setup.prior.distribution
prior_val_mat = Matrix{Float64}(undef, 10_000, length(prior_tuple))
for j = 1:length(prior_tuple)
    prior_val_mat[:, j] .= rand(prior_tuple[j], 10_000)
end
prior_val_mat = col_transformations(prior_val_mat, transformations)
# prior_val_mat[:,11] .= prior_val_mat[:,11].*prior_val_mat[:,4]
# prior_val_mat[:,12] .= prior_val_mat[:,12].*prior_val_mat[:,5]
##
pretty_parameter_names = [
    "Metapop. size dispersion",
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
    "Other. trans. reduction: WHO cng pnt",
]

post_plt = plot(;
    layout = (6, 2),
    size = (800, 2000),
    dpi = 250,
    left_margin = 10mm,
    right_margin = 10mm,
)

for j = 1:length(prior_tuple)
    histogram!(
        post_plt[j],
        val_mat[:, j],
        norm = :pdf,
        fillalpha = 0.5,
        nbins = 100,
        lw = 0.5,
        alpha = 0.1,
        lab = "",
        color = 1,
        title = string(pretty_parameter_names[j]),
    )
    histogram!(
        post_plt[j],
        prior_val_mat[:, j],
        norm = :pdf,
        fillalpha = 0.5,
        alpha = 0.1,
        color = 2,
        nbins = 100,
        lab = "",
    )
    density!(post_plt[j], val_mat[:, j], lw = 3, color = 1, lab = "Posterior")
    density!(post_plt[j], prior_val_mat[:, j], lw = 3, color = 2, lab = "Prior")
end
display(post_plt)
savefig(post_plt, "posteriors/post_plot" * string(wks[end]) * ".png")

##
crn_plt = corner(chn, size = (2000, 2000), left_margin = 5mm, right_margin = 5mm)
savefig(crn_plt, "posteriors/post_crnplot" * string(wks[end]) * ".pdf")

##
