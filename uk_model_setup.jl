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


##

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
    λ = (((N_msm / N_total) * γ_eff * R0_other * total_I) .+( p_trans .* (Λ * I * B))) ./ (N_grp_msm .+ 10)
    λ_other = γ_eff * R0_other * total_I / N_total
    if any(λ .< 0)
        print("Neg inf rate")
    end
    #number of events
    try
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
    catch
        print(I)
    end
    return nothing
end





## Set up for ABC

ingroup = 0.99
n_cliques = 50
ts = wks .|> d -> d - Date(2021, 12, 31) .|> t -> t.value
constants = [N_uk, N_msm, ps, mean_daily_cnts, ingroup, ts, 0.8,n_cliques]

function mpx_sim_function(params, constants, wkly_cases)
    #Get constant data
    N_total, N_msm, ps, ms, ingroup, ts, α_incubation,n_cliques = constants
    #Get parameters and make transformations
    α_choose, p_detect, mean_inf_period, p_trans, R0_other, M, init_scale = params
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
    # observed_cases = zeros(Int64, length(wkly_cases))
    # M = 10.0
    while L1_rel_err < 1.25 && wk_num <= length(wkly_cases) #Step forward weeks and add error kill if rel. L1 error goes above 1.5
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