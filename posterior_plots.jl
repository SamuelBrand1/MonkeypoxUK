using Distributions, StatsBase, StatsPlots
using LinearAlgebra, RecursiveArrayTools
using OrdinaryDiffEq, ApproxBayes, CSV, DataFrames
using JLD2, MCMCChains
using MonkeypoxUK

## Grab UK data and setup model
include("mpxv_datawrangling_inff.jl");
include("setup_model.jl");

## Comment out to use latest data rather than reterospective data

colname = "seqn_fit4"
inferred_prop_na_msm = past_mpxv_data_inferred[:, colname] |> x -> x[.~ismissing.(x)]
mpxv_wkly =
    Matrix(past_mpxv_data_inferred[1:size(inferred_prop_na_msm, 1), ["gbmsm", "nongbmsm"]]) .+
    Vector(past_mpxv_data_inferred[1:size(inferred_prop_na_msm, 1), "na_gbmsm"]) .*
    hcat(inferred_prop_na_msm, 1.0 .- inferred_prop_na_msm)
wks = Date.(past_mpxv_data_inferred.week[1:size(mpxv_wkly, 1)], DateFormat("dd/mm/yyyy"))


##Load posterior draws and structure

smc = MonkeypoxUK.load_smc("posteriors/smc_posterior_draws_2022-09-05_binom.jld2")
param_draws = [part.params for part in smc.particles]

##Create transformations to more interpetable parameters
param_names = [:metapop_size_dispersion, :prob_detect, :mean_inf_period, :prob_transmission,
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

# val_mat = smc.parameters |> X -> col_transformations(X, transformations) |> X -> hcat(X[:,1:10],X[:,11].*X[:,4],X[:,12].*X[:,5])  |> X -> [X[i, j] for i = 1:size(X, 1), j = 1:size(X, 2), k = 1:1]
val_mat = smc.parameters |> X -> col_transformations(X, transformations) |> X -> [X[i, j] for i = 1:size(X, 1), j = 1:size(X, 2), k = 1:1]

chn = Chains(val_mat, param_names)

CSV.write("posteriors/posterior_chain_" * string(wks[end]) * ".csv", DataFrame(chn))

##Calculate orignal R₀ and latest R(t)
function construct_next_gen_mat(params, constants, susceptible_prop, vac_rates)
    
    #Get parameters 
    α_choose, p_detect, mean_inf_period, p_trans, R0_other, M, init_scale, chp_t, trans_red, trans_red_other, scale_trans_red2, scale_red_other2 = params
    #Get constant data
    N_total, N_msm, ps, ms, ingroup, ts, α_incubation, n_cliques, wkly_vaccinations, vac_effectiveness, chp_t2 = constants

    #Calculate next gen matrix
    _A = (ms .* (susceptible_prop .+  (vac_rates .* (1.0 .- vac_effectiveness)))') .* mean_inf_period .* p_trans #Sexual transmission within MSM
    A = _A .+ (R0_other/N_uk) .* repeat(ps' .* N_msm,10) #Other routes of transmission MSM -> MSM
    B = (R0_other*(N_uk - N_msm)/N_total) .* ones(10)    # MSM transmission to non MSM
    C = (R0_other/N_uk) .* ps' .* N_msm  #Non-msm transmission to MSM
    D = [ (R0_other*(N_uk - N_msm)/N_total) ]# Non-MSM transmission to non-MSM
    G = [A B;C D]
    return Real(eigvals(G)[end]), G
end

function generate_Rt_estimate(params, constants, wkly_cases)
    #Get constant data
    N_total, N_msm, ps, ms, ingroup, ts, α_incubation, n_cliques, wkly_vaccinations, vac_effectiveness, chp_t2 = constants

    #Get parameters and make transformations
    α_choose, p_detect, mean_inf_period, p_trans, R0_other, M, init_scale, chp_t, trans_red, trans_red_other, scale_trans_red2, scale_red_other2 = params
    p_γ = 1 / (1 + mean_inf_period)
    γ_eff = -log(1 - p_γ) #get recovery rate
    trans_red2 = trans_red * scale_trans_red2
    trans_red_other2 = scale_trans_red2 * scale_red_other2

    #Generate random population structure
    u0_msm, u0_other, N_clique, N_grp_msm = MonkeypoxUK.setup_initial_state(N_total, N_msm, α_choose, p_detect, α_incubation, ps, init_scale; n_states=9, n_cliques=n_cliques)
    Λ, B = MonkeypoxUK.setup_transmission_matrix(ms, ps, N_clique; ingroup=ingroup)

    #Simulate and track error
    L1_rel_err = 0.0
    total_cases = sum(wkly_cases[1:(end-1), :])
    u_mpx = ArrayPartition(u0_msm, u0_other)
    prob = DiscreteProblem((du, u, p, t) -> MonkeypoxUK.f_mpx_vac(du, u, p, t, Λ, B, N_msm, N_grp_msm, N_total),
        u_mpx, (ts[1] - 7, ts[1] - 7 + 7 * size(wkly_cases, 1)),#lag for week before detection
        [p_trans, R0_other, γ_eff, α_incubation, vac_effectiveness])
    mpx_init = init(prob, FunctionMap(), save_everystep=false) #Begins week 1
    old_onsets = [0, 0]
    new_onsets = [0, 0]
    wk_num = 1
    detected_cases = zeros(size(wkly_cases))
    not_changed = true
    not_changed2 = true

    p_trans_now = mpx_init.p[1]
    R0_other_now = mpx_init.p[2]

    # Get initial state
    u = deepcopy(mpx_init.u)
    S = u.x[1][1, :, :]
    I = u.x[1][6, :, :]
    V = u.x[1][8, :, :]
    S_other = u.x[2][1]
    I_other = u.x[2][6]

    # Transmission rate now instantaneous R(t)
    total_I = I_other + sum(I)
    λ = (p_trans_now .* (Λ * I * B)) ./ (N_grp_msm .+ 1e-5)
    λ_other = γ_eff * R0_other_now * total_I / N_total

    total_inf_rate = sum((S .* (1 .- exp.(-(λ .+ λ_other)))) .+ (V .* (1 .- exp.(-(1 - vac_effectiveness) * (λ .+ λ_other))))) 
    total_inf_rate += S_other * (1 - exp(-λ_other))
    R_0 = total_I > 0 ? mean_inf_period * total_inf_rate / total_I : 0.0



    while wk_num <= size(wkly_cases, 1)
        if not_changed && mpx_init.t > chp_t ##Change point for transmission
            not_changed = false
            mpx_init.p[1] = mpx_init.p[1] * (1 - trans_red) #Reduce transmission per sexual contact after the change point
            mpx_init.p[2] = mpx_init.p[2] * (1 - trans_red_other) #Reduce transmission per non-sexual contact after the change point
        end
        if not_changed2 && mpx_init.t > chp_t2 ##2nd change point for transmission 
            not_changed2 = false
            mpx_init.p[1] = mpx_init.p[1] * (1 - trans_red2) #Reduce sexual MSM transmission after the change point
            mpx_init.p[2] = mpx_init.p[2] * (1 - trans_red_other2) #Reduce  other transmission after the change point
        end
        step!(mpx_init, 7)#Step forward a week

        #Do vaccine uptake
        nv = wkly_vaccinations[wk_num]#Mean number of vaccines deployed
        du_vac = deepcopy(mpx_init.u)
        vac_rate = nv .* du_vac.x[1][1, 3:end, :] / (sum(du_vac.x[1][1, 3:end, :]) .+ 1e-5)
        num_vaccines = map((μ, maxval) -> min(rand(Poisson(μ)), maxval), vac_rate, du_vac.x[1][1, 3:end, :])
        du_vac.x[1][1, 3:end, :] .-= num_vaccines
        du_vac.x[1][8, 3:end, :] .+= num_vaccines
        set_u!(mpx_init, du_vac) #Change the state of the model

        #Calculate actual onsets, generate observed cases and score errors        
        new_onsets = [sum(mpx_init.u.x[1][end, :, :]), mpx_init.u.x[2][end]]
        actual_obs = [rand(BetaBinomial(new_onsets[1] - old_onsets[1], p_detect * M, (1 - p_detect) * M)), rand(BetaBinomial(new_onsets[2] - old_onsets[2], p_detect * M, (1 - p_detect) * M))]
        detected_cases[wk_num, :] .= actual_obs #lag 1 week

        wk_num += 1
        old_onsets = new_onsets
    end
    
    #Calculate final R(t)
    p_trans_now = mpx_init.p[1]
    R0_other_now = mpx_init.p[2]

    # Get final state
    u = deepcopy(mpx_init.u)
    S = u.x[1][1, :, :]
    I = u.x[1][6, :, :]
    V = u.x[1][8, :, :]
    S_other = u.x[2][1]
    I_other = u.x[2][6]

    # Transmission rate now instantaneous R(t)
    total_I = I_other + sum(I)
    λ = (p_trans_now .* (Λ * I * B)) ./ (N_grp_msm .+ 1e-5)
    λ_other = γ_eff * R0_other_now * total_I / N_total

    total_inf_rate = sum((S .* (1 .- exp.(-(λ .+ λ_other)))) .+ (V .* (1 .- exp.(-(1 - vac_effectiveness) * (λ .+ λ_other))))) 
    total_inf_rate += S_other * (1 - exp(-λ_other))
    R_t = total_I > 0 ? mean_inf_period * total_inf_rate / total_I : 0.0

    return R_t, R_0, S, I, p_trans_now, R0_other_now
end

## Calculate the original R0 with next gen matrix method and lastest R(t)
R0s = map(θ -> construct_next_gen_mat(θ,constants, [ones(10); zeros(0)], [zeros(10);fill(1.0,0)])[1],param_draws )
initial_Rts = map(θ -> generate_Rt_estimate(θ, constants, mpxv_wkly)[2], param_draws)

latest_Rts = map(θ -> generate_Rt_estimate(θ, constants, mpxv_wkly)[1], param_draws)

@show round(mean(R0s),digits = 2),round.(quantile(R0s,[0.1,0.9]),digits = 2)
@show round(mean(latest_Rts), digits = 2), round.(quantile(latest_Rts,[0.1,0.9]), digits = 2)
@show round(mean(initial_Rts), digits = 2), round.(quantile(initial_Rts[initial_Rts .> 1.0],[0.1,0.9]), digits = 2)



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
pretty_parameter_names = ["Metapop. size dispersion",
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