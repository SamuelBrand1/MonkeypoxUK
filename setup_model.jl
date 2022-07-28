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

d_incubation = Gamma(6.77, 1 / 0.77)#Fit from rerunning 
# d_incubation = LogNormal(2.09, 0.46)
# d_incubation = Gamma(7, 1.25)
negbin_std = fill(0.0, 8)
for r = 1:8
    p = r / mean(d_incubation)
    negbin_std[r] = std(NegativeBinomial(r, p))
end
plt_incfit = bar(negbin_std,
    title = "Discrete time model vs data-driven model for incubation",
    lab = "",
    xticks = 1:8,
    xlabel = "Number of stages",
    ylabel = "Std. incubation (days)",
    size = (800,600),left_margin = 5mm)
hline!(plt_incfit,[std(d_incubation)],lab = "std. data-driven model")
display(plt_incfit)
savefig(plt_incfit,"plots/incubation_fit.png")
#Optimal choice is 4 stages with effective rate to match the mean
p_incubation = 4 / mean(d_incubation)
α_incubation_eff = -log(1 - p_incubation)
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

function setup_initial_state(N_pop, N_msm, α_choose, p_detect, γ_eff, ps, init_scale; n_states=7, n_cliques=50)
    u0_msm = zeros(Int64, n_states, length(ps), n_cliques)
    u0_other = zeros(Int64, n_states)
    N_clique = rand(DirichletMultinomial(N_msm, α_choose * ones(n_cliques)))
    for k = 1:n_cliques
        u0_msm[1, :, k] .= rand(Multinomial(N_clique[k], ps))
    end
    #Add infecteds so that expected detections on week 1 are 1
    choose_clique = rand(Categorical(normalize(N_clique, 1)))
    av_infs = 5 * init_scale / (p_detect * (1 - exp(-γ_eff))) #Set av number of infecteds across 5 categories of incubation and infectious rescaled by daily probability of detection if infectious
    u0_msm[2:6, :, choose_clique] .= map(μ -> rand(Poisson(μ)), fill(av_infs / (5 * length(ps)), 5, length(ps)))
    u0_msm = u0_msm[:, :, N_clique.>0] #Reduce group size
    #Set up non MSM population
    u0_other[1] = N_pop
    N_grp_msm = u0_msm[1, :, :]
    return u0_msm, u0_other, N_clique[N_clique.>0], N_grp_msm
end

@time u0_msm, u0_other, N_clique, N_grp_msm = setup_initial_state(N_uk, N_msm, 0.11, 0.5, 1 / 7, ps, 2.0)

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
    E = @view u.x[1][2:5, :, :]
    I = @view u.x[1][6, :, :]
    R = @view u.x[1][7, :, :]
    S_other = u.x[2][1]
    E_other = @view u.x[2][2:5]
    I_other = u.x[2][6]
    R_other = u.x[2][7]

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
    du.x[1][2:5, :, :] .-= num_incs
    du.x[1][3:6, :, :] .+= num_incs
    du.x[2][2:5] .-= num_incs_other
    du.x[2][3:6] .+= num_incs_other
    #recoveries
    du.x[1][6, :, :] .-= num_recs
    du.x[1][7, :, :] .+= num_recs
    du.x[2][6] -= num_recs_other
    du.x[2][7] += num_recs_other

    return nothing
end

function f_mpx_vac(du, u, p, t, Λ, B, N_msm, N_grp_msm, N_total)
    p_trans, R0_other, γ_eff, α_incubation, vac_eff = p

    #states
    S = @view u.x[1][1, :, :]
    E = @view u.x[1][2:8, :, :]
    I = @view u.x[1][9, :, :]
    R = @view u.x[1][10, :, :]
    V = @view u.x[1][11, :, :]

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
    num_infs_vac = map((n, p) -> rand(Binomial(n, p)), V, 1 .- exp.(-(1 - vac_eff) * (λ .+ λ_other)))#infections among MSM with vaccine
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
    du.x[1][11, :, :] .-= num_infs_vac
    du.x[1][2, :, :] .+= num_infs .+ num_infs_vac
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
constants = [N_uk, N_msm, ps, mean_daily_cnts, ingroup, ts, α_incubation_eff, n_cliques]


function mpx_sim_function_chp(params, constants, wkly_cases)
    #Get constant data
    N_total, N_msm, ps, ms, ingroup, ts, α_incubation, n_cliques = constants
    #Get parameters and make transformations
    α_choose, p_detect, mean_inf_period, p_trans, R0_other, M, init_scale, chp_t, trans_red, trans_red_other = params
    p_γ = 1 / (1 + mean_inf_period)
    γ_eff = -log(1-p_γ) #get recovery rate
    # M = (1/ρ) + 1 #effective sample size for Beta-Binomial
    #Generate random population structure
    u0_msm, u0_other, N_clique, N_grp_msm = setup_initial_state(N_total, N_msm, α_choose, p_detect, γ_eff, ps, init_scale; n_cliques=n_cliques)
    Λ, B = setup_transmission_matrix(ms, ps, N_clique; ingroup=ingroup)

    #Simulate and track error
    L1_rel_err = 0.0
    total_cases = sum(wkly_cases[3:end-1,:])
    u_mpx = ArrayPartition(u0_msm, u0_other)
    prob = DiscreteProblem((du, u, p, t) -> f_mpx(du, u, p, t, Λ, B, N_msm, N_grp_msm, N_total),
        u_mpx, (ts[1], ts[end]),
        [p_trans, R0_other, γ_eff, α_incubation])
    mpx_init = init(prob, FunctionMap()) #Begins week 1
    old_recs = [0, 0]
    new_recs = [0, 0]
    wk_num = 1
    detected_cases = zeros(size(wkly_cases))
    not_changed = true

    while wk_num <= size(wkly_cases, 1) 
        if not_changed && mpx_init.t > chp_t ##Change point for transmission
            not_changed = false
            mpx_init.p[1] = mpx_init.p[1] * (1 - trans_red) #Reduce transmission per sexual contact after the change point
            mpx_init.p[2] = mpx_init.p[2] * (1 - trans_red_other) #Reduce transmission per non-sexual contact after the change point
        end
        step!(mpx_init, 7)#Step forward a week
        new_recs = [sum(mpx_init.u.x[1][end, :, :]), mpx_init.u.x[2][end]]
        actual_obs = [rand(BetaBinomial(new_recs[1] - old_recs[1], p_detect * M, (1 - p_detect) * M)), rand(BetaBinomial(new_recs[2] - old_recs[2], p_detect * M, (1 - p_detect) * M))]
        detected_cases[wk_num,:] .= actual_obs
        if wk_num > 3 && wk_num < length(wkly_cases) # Only compare on weeks 3 --- (end-1)
            L1_rel_err += sum(abs, actual_obs .- wkly_cases[wk_num,:]) / total_cases
        end
        wk_num += 1
        old_recs = new_recs
    end

    return L1_rel_err, detected_cases
end





function mpx_sim_function_interventions(params, constants, wkly_cases, interventions)
    #Get constant data
    N_total, N_msm, ps, ms, ingroup, ts, α_incubation, n_cliques = constants
    #Get intervention data
    chp_t2 = interventions.chp_t2
    wkly_vaccinations = interventions.wkly_vaccinations
    trans_red2 = interventions.trans_red2
    inf_duration_red = interventions.inf_duration_red
    vac_effectiveness = interventions.vac_effectiveness
    R0_other_red = interventions.R0_other_red


    #Get parameters and make transformations
    α_choose, p_detect, mean_inf_period, p_trans, R0_other, M, init_scale, chp_t, trans_red = params
    γ_eff = 1 / mean_inf_period #get recovery rate

    #Generate random population structure
    u0_msm, u0_other, N_clique, N_grp_msm = setup_initial_state(N_total, N_msm, α_choose, p_detect, γ_eff, ps, init_scale; n_states=11, n_cliques=n_cliques)
    Λ, B = setup_transmission_matrix(ms, ps, N_clique; ingroup=ingroup)

    #Simulate and track error
    L1_rel_err = 0.0
    total_cases = sum(wkly_cases[3:end-1])
    u_mpx = ArrayPartition(u0_msm, u0_other)
    prob = DiscreteProblem((du, u, p, t) -> f_mpx_vac(du, u, p, t, Λ, B, N_msm, N_grp_msm, N_total),
        u_mpx, (ts[1], ts[1] + 7 * length(wkly_cases)),
        [p_trans, R0_other, γ_eff, α_incubation, vac_effectiveness])
    mpx_init = init(prob, FunctionMap(), save_everystep=false) #Begins week 1
    old_recs = [0, 0]
    new_recs = [0, 0]
    old_sus = [sum(u0_msm[1, :, :]), u0_other[1]]
    new_sus = [sum(u0_msm[1, :, :]), u0_other[1]]
    wk_num = 1
    detected_cases = zeros(length(wkly_cases))
    incidence = zeros(length(wkly_cases))

    not_changed = true
    not_changed2 = true


    while wk_num <= length(wkly_cases) #Step forward a week
        if not_changed && mpx_init.t > chp_t ##1st change point for transmission prob
            not_changed = false
            mpx_init.p[1] = mpx_init.p[1] * (1 - trans_red) #Reduce transmission after the change point
        end
        if not_changed2 && mpx_init.t > chp_t2 ##2nd change point for transmission 
            not_changed2 = false
            mpx_init.p[1] = mpx_init.p[1] * (1 - trans_red2) #Reduce transmission after the change point
            mpx_init.p[2] = mpx_init.p[2] * (1 - R0_other_red) #Reduce transmission after the change point
            mpx_init.p[3] = 1 / (mean_inf_period * (1 - inf_duration_red)) #Reduce duration of transmission after the change point
        end
        #Step forward a week in time
        step!(mpx_init, 7)
        #Do vaccine uptake
        nv = wkly_vaccinations[wk_num]#Mean number of vaccines deployed
        du_vac = deepcopy(mpx_init.u)
        vac_rate = nv .* du_vac.x[1][1, 3:end, :] / sum(du_vac.x[1][1, 3:end, :])
        num_vaccines = map((μ, maxval) -> min(rand(Poisson(μ)), maxval), vac_rate, du_vac.x[1][1, 3:end, :])
        du_vac.x[1][1, 3:end, :] .-= num_vaccines
        du_vac.x[1][11, 3:end, :] .+= num_vaccines
        set_u!(mpx_init, du_vac) #Change the state of the model
        #Calculate actual recoveries and score errors
        new_recs = [sum(mpx_init.u.x[1][10, :, :]), mpx_init.u.x[2][10]]
        new_sus = [sum(mpx_init.u.x[1][1, :, :]), mpx_init.u.x[2][1]]
        actual_obs = [rand(BetaBinomial(new_recs[1] - old_recs[1], p_detect * M, (1 - p_detect) * M)), rand(BetaBinomial(new_recs[2] - old_recs[2], p_detect * M, (1 - p_detect) * M))]
        detected_cases[wk_num] = Float64.(sum(actual_obs))
        incidence[wk_num] = Float64.(sum(old_sus .- new_sus) - sum(num_vaccines))
        if wk_num > 3 && wk_num < length(wkly_cases) # Only compare on weeks 3 --- (end-1)
            L1_rel_err += sum(abs, actual_obs .- wkly_cases[wk_num] .* [0.99, 0.01]) / total_cases
        end
        wk_num += 1
        old_recs = new_recs
        old_sus = new_sus

    end

    return L1_rel_err, detected_cases, incidence
end

##Posterior draws

