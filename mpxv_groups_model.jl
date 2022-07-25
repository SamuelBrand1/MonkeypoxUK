## Idea is to have both fitness and SBM effects in sexual contact

using Distributions, StatsBase
using LinearAlgebra, RecursiveArrayTools
using OrdinaryDiffEq, ApproxBayes
using JLD2, MLUtils

## Grab UK data

include("mpxv_datawrangling.jl");

## Group size

N_uk = 67.22e6
prop_ovr18 = 0.787
prop_msm = 0.034 #https://www.ons.gov.uk/peoplepopulationandcommunity/culturalidentity/sexuality/bulletins/sexualidentityuk/2020
prop_sexual_active = 1 - 0.154 #A dynamic power-law sexual network model of gonorrhoea outbreaks

N_msm = round(Int64, N_uk * prop_ovr18 * prop_msm * prop_sexual_active) #~1.5m


## Incubation period

# Using  this  best-fitting  distribution,  the  mean  incuba-tion period was estimated to be 8.5 days
#  (95% credible intervals (CrI): 6.6–10.9 days), 
# with the 5th percentile of 4.2 days and the 95th percentile of 17.3 days (Table 2)

# d_incubation = Gamma(6.77,1/0.77)#Fit from rerunning 
d_incubation = Gamma(7, 1.25)#

mean(d_incubation), quantile(d_incubation, 0.05),
quantile(d_incubation, 0.95)

# d_incubation = LogNormal(2.09,0.46)#<--- from running the code and using posterior mean
# mean(d_incubation),quantile(d_incubation,0.05),
#     quantile(d_incubation,0.95)
d_infectious = Gamma(1, 7 / 1)
p_inc = [cdf(d_incubation, t) - cdf(d_incubation, t - 1) for t = 1:60]
Q_inf = [ccdf(d_infectious, t) for t = 1:60]
p_inf = [sum(p_inc[1:(t-1)] .* Q_inf[(t-1):-1:1]) for t = 1:60]
w = normalize(p_inf, 1)
bar(p_inf)
r = log(3.5) / 55
R0 = 1 / sum([exp(-r * t) * w[t] for t = 1:60])

## Set up equipotential groups

n_grp = 10
α_scaling = 0.82
x_min = 1.0
x_max = 3650.0

#Mean rate of yearly contacts over all MSMS
X̄ = (α_scaling / (α_scaling - 1)) * (x_min^(1 - α_scaling) - x_max^(1 - α_scaling)) / (x_min^(-α_scaling) - x_max^(-α_scaling))

#Calculation

C = (x_min^(-α_scaling) - x_max^(-α_scaling)) * X̄ / n_grp * ((1 - α_scaling) / α_scaling)
xs = zeros(n_grp + 1)
xs[1] = x_min
for k = 2:n_grp
    xs[k] = (xs[k-1]^(1 - α_scaling) + C)^(1 / (1 - α_scaling))
end
xs[end] = x_max

#Percentages in each group

ps = map(x -> (x_min^(-α_scaling) - x^(-α_scaling)) / (x_min^(-α_scaling) - x_max^(-α_scaling)), xs) |> diff

#Mean daily contact rates within groups

xs_pairs = [(xs[i], xs[i+1]) for i = 1:(length(xs)-1)]
mean_daily_cnts = map(x -> (α_scaling / (α_scaling - 1)) * (x[1]^(1 - α_scaling) - x[2]^(1 - α_scaling)) / (x[1]^(-α_scaling) - x[2]^(-α_scaling)), xs_pairs) .|> x -> x / 365.25

##

function setup_initial_state(N_pop, N_msm, α_choose, p_detect, γ_eff, ps; n_states=10, n_cliques=50)
    u0_msm = zeros(Int64, n_states, length(ps), n_cliques)
    u0_other = zeros(Int64, n_states)
    N_clique = rand(DirichletMultinomial(N_msm, α_choose * ones(n_cliques)))
    for k = 1:n_cliques
        u0_msm[1, :, k] .= rand(Multinomial(N_clique[k], ps))
    end
    #add infecteds so that expected detections on week 1 are 1
    choose_clique = rand(Categorical(normalize(N_clique, 1)))
    n_infs = 8 * ceil(Int64, 1 / (p_detect * (1 - exp(-γ_eff))))
    u0_msm[2:9, :, choose_clique] .= map(μ -> rand(Poisson(μ)), fill(n_infs / (3.5 * 8 * length(ps)), 8, length(ps)))
    #
    u0_other[1] = N_pop
    N_grp_msm = u0_msm[1, :, :]
    return u0_msm, u0_other, N_clique, N_grp_msm
end

@time u0_msm, u0_other, N_clique, N_grp_msm = setup_initial_state(N_uk, N_msm, 0.1, 0.5, 1 / 7, ps)

function setup_transmission_matrix(ms, ps, N_clique; ingroup=0.99)
    n_cliques = length(N_clique)
    n_grps = length(ps)
    B = Matrix{Float64}(undef, n_cliques, n_cliques) #Between group contacting
    for i = 1:n_cliques, j = 1:n_cliques
        if i == j
            B[i, i] = ingroup
        else
            B[i, j] = (1 - ingroup) * (N_clique[j] / (sum(N_clique) - N_clique[i]))
        end
    end
    Λ = Matrix{Float64}(undef, n_grps, n_grps)
    attraction_vect = ms .* ps ./ sum(ms .* ps)
    for i = 1:n_grps, j = 1:n_grps
        Λ[i, j] = attraction_vect[i] * ms[j]
    end
    return Λ, B
end

@time Λ, B = setup_transmission_matrix(mean_daily_cnts, ps, N_clique; ingroup=0.99)




function f_mpx(du, u, p, t, Λ, B, N_msm, N_grp_msm, N_total)
    p_trans, R0_other, γ_eff, α_incubation = p

    #states
    S = @view u.x[1][1, :, :]
    E = @view u.x[1][2:8, :, :]
    I = @view u.x[1][9, :, :]
    R = @view u.x[1][10, :, :]
    S_other = u.x[2][1]
    E_other = @view u.x[2][2:8]
    I_other = u.x[2][9]
    R_other = u.x[2][10]

    #force of infection
    total_I = I_other + sum(I)
    λ = ((N_msm / N_total) * γ_eff * R0_other * total_I .+ p_trans .* Λ * I * B) ./ (N_grp_msm .+ 1e-5)
    λ_other = γ_eff * R0_other * total_I / N_total

    #number of events
    num_infs = map((n, p) -> rand(Binomial(n, p)), S, 1 .- exp.(-λ))#infections among MSM
    num_incs = map(n -> rand(Binomial(n, 1 - exp(-α_incubation))), E)#incubation among MSM
    num_recs = map(n -> rand(Binomial(n, 1 - exp(-γ_eff))), I)#recovery among MSM
    num_infs_other = rand(Binomial(S_other, 1 - exp(-λ_other)))#infections among non MSM
    num_incs_other = map(n -> rand(Binomial(n, 1 - exp(-α_incubation))), E_other)#incubation among non MSM
    num_recs_other = rand(Binomial(I_other, 1 - exp(-γ_eff)))#recovery among non MSM

    #create change
    du.x[1] .= u.x[1]
    du.x[2] .= u.x[2]
    #infections
    du.x[1][1, :, :] .-= num_infs
    du.x[1][2, :, :] .+= num_infs
    du.x[2][1] -= num_infs_other
    du.x[2][2] += num_infs_other
    #incubations
    du.x[1][2:8, :, :] .-= num_incs
    du.x[1][3:9, :, :] .+= num_incs
    du.x[2][2:8] .-= num_incs_other
    du.x[2][3:9] .+= num_incs_other
    #recoveries
    du.x[1][9, :, :] .-= num_recs
    du.x[1][10, :, :] .+= num_recs
    du.x[2][9] -= num_recs_other
    du.x[2][10] += num_recs_other

    return nothing
end

u_mpx = ArrayPartition(u0_msm, u0_other)
du_mpx = similar(u_mpx)

@time f_mpx(du_mpx, u_mpx, rand(4), 0, Λ, B, N_msm, N_grp_msm, N_uk)
du_mpx.x[2]


ts = wks .|> d -> d - Date(2021, 12, 31) .|> t -> t.value

prob = DiscreteProblem((du, u, p, t) -> f_mpx(du, u, p, t, Λ, B, N_msm, N_grp_msm, N_uk),
    u_mpx, (ts[1], ts[end]), [0.1, 0.9, 1 / 7, 0.8])

mpx_init = init(prob, FunctionMap())
step!(mpx_init, 7)

@time sol = solve(prob, FunctionMap(), saveat=7)
Rs = [sum(u.x[1][10, :, :]) + u.x[2][end] for u in sol.u]


## Set up for ABC
ingroup = 0.99
constants = [N_uk, N_msm, ps, mean_daily_cnts, ingroup, ts, 0.8]

function mpx_sim_function(params, constants, wkly_cases)
    #Get constant data
    N_total, N_msm, ps, ms, ingroup, ts, α_incubation = constants
    #Get parameters and make transformations
    α_choose, p_detect, mean_inf_period, p_trans, R0_other, M = params
    γ_eff = 1 / mean_inf_period #get recovery rate
    # M = (1/ρ) + 1 #effective sample size for Beta-Binomial
    #Generate random population structure
    u0_msm, u0_other, N_clique, N_grp_msm = setup_initial_state(N_total, N_msm, α_choose, p_detect, γ_eff, ps)
    Λ, B = setup_transmission_matrix(ms, ps, N_clique; ingroup=ingroup)

    #Simulate and track error
    L1_rel_err = 0.0

    total_cases = sum(wkly_cases[3:end-1])
    u_mpx = ArrayPartition(u0_msm, u0_other)
    prob = DiscreteProblem((du, u, p, t) -> f_mpx(du, u, p, t, Λ, B, N_msm, N_grp_msm, N_total),
        u_mpx, (ts[1], ts[end]),
        [p_trans, R0_other, γ_eff, α_incubation])
    mpx_init = init(prob, FunctionMap()) #Begins week 1
    old_recs = [0, 0]
    new_recs = [0, 0]
    wk_num = 1
    detected_cases = zeros(length(wkly_cases))
    observed_cases = zeros(Int64, length(wkly_cases))

    try
        for i = 1:2 #Step forward 2 weeks (not used in fitting because early data)
            step!(mpx_init, 7)
            new_recs = [sum(mpx_init.u.x[1][10, :, :]), mpx_init.u.x[2][10]]
            detected_cases[wk_num] = (sum(new_recs) - sum(old_recs)) * p_detect
            wk_num += 1
            old_recs = new_recs
        end

        while L1_rel_err < 1.0 && wk_num < length(wkly_cases) #Step forward weeks and add error kill if rel. L1 error goes above 1
            step!(mpx_init, 7)
            new_recs = [sum(mpx_init.u.x[1][10, :, :]), mpx_init.u.x[2][10]]
            detected_cases[wk_num] = (sum(new_recs) - sum(old_recs)) * p_detect
            actual_obs = [rand(BetaBinomial(new_recs[1] - old_recs[1], p_detect * M, (1 - p_detect) * M)), rand(BetaBinomial(new_recs[2] - old_recs[2], p_detect * M, (1 - p_detect) * M))]
            # observed_cases[wk_num] = sum(actual_obs)
            L1_rel_err += sum(abs, actual_obs .- wkly_cases[wk_num] .* [0.99, 0.01]) / total_cases
            wk_num += 1
            old_recs = new_recs
        end
        # Last step for comparison purposes
        step!(mpx_init, 7)
        new_recs = [sum(mpx_init.u.x[1][10, :, :]), mpx_init.u.x[2][10]]
        detected_cases[wk_num] = (sum(new_recs) - sum(old_recs)) * p_detect
        return L1_rel_err, detected_cases
    catch
        return Inf, 0
    end
end

##


_p = [0.01, 0.75, 7, 0.25, 0.8, 0.1]
err, pred = mpx_sim_function(_p, constants, mpxv_wkly)

plot(pred)
scatter!(mpxv_wkly)

## Priors

# α_choose,p_detect,1/γ_eff,p_trans,R0_other,ρ
prior_vect = [Gamma(2, 1 / 2), Beta(5, 5), Gamma(3, 7 / 3), Beta(10, 90), LogNormal(0, 0.5), Gamma(2, 50 / 2)]


##run ABC

setup = ABCSMC(mpx_sim_function, #simulation function
    6, # number of parameters
    0.1, #target ϵ
    Prior(prior_vect); #Prior for each of the parameters
    ϵ1=100.0,
    convergence=0.05,
    nparticles=2000,
    kernel=gaussiankernel,
    constants=constants,
    maxiterations=10^9)

smc = runabc(setup, mpxv_wkly, verbose=true, progress=true, parallel=true)

##
@save("results.jld2", smc)
# preds = MLUtils.stack([part.other for part in smc.particles],dims = 1)
##

preds = [part.other for part in smc.particles]
plt = plot()
for pred in preds
    plot!(plt, pred, alpha=0.1, lw=2, color=:grey, lab="")
end
scatter!(plt, 3:10, mpxv_wkly[3:(end-1)], lab="data (used in fitting)", color=:red)
scatter!(plt, [1, 2, 11], mpxv_wkly[[1, 2, 11]], lab="data (not in fitting)", color=:blue)
display(plt)

## 
@load(["results.jld2"])

##
posterior_draws_distrib =
    [Normal(0.68, 0.63 / 3),
        Normal(0.25, 0.17 / 3),
        Normal(12.48, 8.3 / 3),
        Normal(0.09, 0.03 / 3),
        Normal(0.37, 0.12 / 3),
        Normal(8.84, 7 / 3)]

posterior_draws = [clamp!(rand.(posterior_draws_distrib), 0.01, Inf) for k = 1:2000]
mpx_sim_function(posterior_draws[1], constants, mpxv_wkly)[2]

preds = [mpx_sim_function(draw, constants, mpxv_wkly)[2] for draw in posterior_draws]
preds = filter(x -> x[end] > 0, preds)
##
plt = plot(xticks=(1:11, wks),
    ylabel="Weekly reported MPX",
    title="UK MPX",
    size=(1000, 400), dpi=250,
    left_margin=5mm)
for pred in preds
    plot!(plt, pred, alpha=0.1, lw=2, color=:grey, lab="")
end
scatter!(plt, 3:10, mpxv_wkly[3:(end-1)], ms=7, lab="Data (used in fitting)", color=:red)
scatter!(plt, [1, 2, 11], mpxv_wkly[[1, 2, 11]], ms=7, lab="Data (not in fitting)", color=:blue)
display(plt)

