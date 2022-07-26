## Idea is to have both fitness and SBM effects in sexual contact

using Distributions, StatsBase, StatsPlots
using LinearAlgebra, RecursiveArrayTools
using OrdinaryDiffEq, ApproxBayes
using JLD2, MLUtils

## Grab UK data

include("mpxv_datawrangling.jl");

## UK group sizes

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
d_incubation = Gamma(7, 1.25)


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

function setup_initial_state(N_pop, N_msm, α_choose, p_detect, γ_eff, ps,init_scale; n_states=10, n_cliques=50)
    u0_msm = zeros(Int64, n_states, length(ps), n_cliques)
    u0_other = zeros(Int64, n_states)
    N_clique = rand(DirichletMultinomial(N_msm, α_choose * ones(n_cliques)))
    for k = 1:n_cliques
        u0_msm[1, :, k] .= rand(Multinomial(N_clique[k], ps))
    end
    #Add infecteds so that expected detections on week 1 are 1
    choose_clique = rand(Categorical(normalize(N_clique,1)))
    av_infs = 8*init_scale/(p_detect*(1-exp(-γ_eff))) #Set av number of infecteds across 8 categories of incubation and infectious rescaled by daily probability of detection if infectious
    u0_msm[2:9,:,choose_clique] .= map(μ -> rand(Poisson(μ)), fill(av_infs/(8*length(ps)),8,length(ps)))
    u0_msm = u0_msm[:,:,N_clique .>0] #Reduce group size
    #Set up non MSM population
    u0_other[1] = N_pop
    N_grp_msm = u0_msm[1, :, :]
    return u0_msm, u0_other, N_clique[N_clique .>0], N_grp_msm
end

@time u0_msm, u0_other, N_clique, N_grp_msm = setup_initial_state(N_uk, N_msm, 0.0011, 0.5, 1 / 7, ps,2.0)

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

## Transmission dynamics

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
    λ = (p_trans .* (Λ * I * B)) ./ (N_grp_msm .+ 1e-5)
    λ_other = γ_eff * R0_other * total_I / N_total
    #number of events
    
    num_infs = map((n, p) -> rand(Binomial(n, p)), S, 1 .- exp.(-(λ .+ λ_other)))#infections among MSM
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

## Set up for ABC

ingroup = 0.99
n_cliques = 50
ts = wks .|> d -> d - Date(2021, 12, 31) .|> t -> t.value
constants = [N_uk, N_msm, ps, mean_daily_cnts, ingroup, ts, 0.8,n_cliques]


function mpx_sim_function_chp(params, constants, wkly_cases)
    #Get constant data
    N_total, N_msm, ps, ms, ingroup, ts, α_incubation,n_cliques = constants
    #Get parameters and make transformations
    α_choose, p_detect, mean_inf_period, p_trans, R0_other, M, init_scale,chp_t,trans_red = params
    γ_eff = 1 / mean_inf_period #get recovery rate
    # M = (1/ρ) + 1 #effective sample size for Beta-Binomial
    #Generate random population structure
    u0_msm, u0_other, N_clique, N_grp_msm = setup_initial_state(N_total, N_msm, α_choose, p_detect, γ_eff, ps, init_scale;n_cliques = n_cliques)
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
    not_changed = true
    
    while wk_num <= length(wkly_cases) #Step forward a week
        if not_changed && mpx_init.t > chp_t ##Change point for transmission
            not_changed = false
            mpx_init.p[1] = mpx_init.p[1]*(1-trans_red) #Reduce transmission after the change point
        end
        step!(mpx_init,7)
        new_recs = [sum(mpx_init.u.x[1][10,:,:]),mpx_init.u.x[2][10]]
        # detected_cases[wk_num] = (sum(new_recs) - sum(old_recs))*p_detect
        actual_obs = [rand(BetaBinomial(new_recs[1] - old_recs[1],p_detect*M,(1-p_detect)*M)),rand(BetaBinomial(new_recs[2] - old_recs[2],p_detect*M,(1-p_detect)*M))]
        detected_cases[wk_num] = Float64.(sum(actual_obs))
        if wk_num > 3 && wk_num < length(wkly_cases) # Only compare on weeks 3 --- (end-1)
            # L1_rel_err += sum(abs,p_detect.*(new_recs .- old_recs) .- wkly_cases[wk_num].*[0.99,0.01])/total_cases
            L1_rel_err += sum(abs,actual_obs .- wkly_cases[wk_num].*[0.99,0.01])/total_cases
        end
        wk_num += 1
        old_recs = new_recs
    end
    
    return L1_rel_err, detected_cases
end

##
_p = [0.01, 0.5, 20, 0.2, 0.5,10,1.5,182,1.0]

err, pred = mpx_sim_function_chp(_p, constants, mpxv_wkly)

plt = plot(pred)
scatter!(plt,mpxv_wkly)
display(plt)
print(err)

## Priors - chg point model

# α_choose, p_detect, mean_inf_period, p_trans, R0_other, M, init_scale ,chp_t,trans_red
prior_vect_cng_pnt = [Gamma(1, 1), # α_choose 1
                Beta(5, 5), #p_detect  2
                Gamma(3, 10 / 3), #mean_inf_period 3
                Beta(5, 45), #p_trans  4
                LogNormal(log(0.75), 0.25), #R0_other 5
                Gamma(3, 10 / 3),#  M 6
                LogNormal(0,1),#init_scale 7
                Uniform(152,ts[end]),# chp_t 8
                Beta(5,5)]#trans_red 9
## Prior predictive checking - simulation
draws = [rand.(prior_vect_cng_pnt) for i = 1:1000]
prior_sims = map(θ -> mpx_sim_function_chp(θ,constants,mpxv_wkly),draws)
##Prior predictive checking - simulation
prior_preds = [sim[2] for sim in prior_sims]
plt = plot(;ylabel = "Weekly cases",
            title = "Prior predictive checking")
for pred in prior_preds
    plot!(plt,wks,pred,lab = "", color = :grey, alpha = 0.3)
end
display(plt)
savefig(plt,"plots/prior_predictive_checking_plot.png")

## Model-based calibration of target tolerance
min_mbc_errs = map(n -> minimum(map(x -> mpx_sim_function_chp(draws[n],constants,prior_sims[n][2])[1],1:5)),1:1000)


##
err_hist = histogram(min_mbc_errs,norm = :pdf,nbins = 200,
            lab = "",
            title = "Sampled errors from simulations with exact parameters",
            xlabel = "Median L1 relative error",
            size = (700,400))
vline!(err_hist,[0.543],lab = "5th percentile (rel. err. = 0.543)",lw = 3)
display(err_hist)
savefig(err_hist,"plots/mbc_error_calibration_plt.png")
##Run inference

setup_cng_pnt = ABCSMC(mpx_sim_function_chp, #simulation function
    9, # number of parameters
    0.543, #target ϵ
    Prior(prior_vect_cng_pnt); #Prior for each of the parameters
    ϵ1=100,
    convergence=0.05,
    nparticles = 1000,
    α = 0.5,
    kernel=gaussiankernel,
    constants=constants,
    maxiterations=10^10)

smc_cng_pnt = runabc(setup_cng_pnt, mpxv_wkly, verbose=true, progress=true)#, parallel=true)

##
@save("draws2_nrw.jld2",smc_cng_pnt)


##posterior predictive checking - simulation
post_preds = [part.other for part in smc_cng_pnt.particles]
plt = plot(;ylabel = "Weekly cases",
            title = "Prior predictive checking")
for pred in post_preds
    plot!(plt,wks,pred,lab = "", color = :grey, alpha = 0.3)
end
scatter!(plt,wks,mpxv_wkly,lab = "Data")
display(plt)

