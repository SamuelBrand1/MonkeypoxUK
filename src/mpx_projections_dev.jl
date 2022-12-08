generated_quantities = [particle.other for particle in smc.particles]
end_states = [particle.other.end_state for particle in smc.particles]

u_msm = end_states[1].x[1]
N_clique = sum(u_msm, dims = [1,2])[:]

N_grp_msm = sum(u_msm,dims = 1)[1, :, :]

function mpx_sim_function_projections(params, constants, wkly_cases, interventions, init_condition, ts_0, weeks_to_project, weeks_to_reversion)
    #Get constant data
    N_total, N_msm, ps, ms, ingroup, ts, α_incubation, γ_eff, epsilon, n_cliques, wkly_vaccinations, vac_effectiveness, chp_t2, weeks_to_change = constants

    #Get intervention data
    chp_t2 = interventions.chp_t2
    wkly_vaccinations = interventions.wkly_vaccinations
    trans_red2 = interventions.trans_red2
    vac_effectiveness = interventions.vac_effectiveness
    trans_red_other2 = interventions.trans_red_other2

    #Get parameters and make transformations
    α_choose, p_detect, p_trans, R0_other, M, init_scale, chp_t, trans_red, trans_red_other, trans_red2, trans_red_other2 = params
    κ_cng = (weeks_to_change * 7 / 2) / 4.6 # logistic scale for behaviour change to occur over: 4.6 is κ = 1 time to go from 0.01 to 0.5 and 0.5 to 0.99

    #Calculate logistic reversion to pre-change
    p_min = p_trans*(1 - trans_red)*(1 - trans_red2)
    R_oth_min = R0_other*(1 - trans_red_other)*(1 - trans_red_other2)
    T₅₀ = (Date(2022,9,1) - Date(2021,12,31)).value + (weeks_to_reversion * 7 / 2) # 50% return to normal point
    κ_rev = (weeks_to_reversion * 7 / 2) / 4.6 # logistic scale for return to normal: 4.6 is κ = 1 time to go from 0.01 to 0.5 and 0.5 to 0.99

    # wkly_reversion = exp(-(log(1 - trans_red) + log(1 - trans_red2))/weeks_to_reversion)
    # wkly_reversion_othr = exp(-(log(1 - trans_red_other) + log(1 - trans_red_other2))/weeks_to_reversion)
    # Generate transmission matrices based on the inital state
    N_clique = sum(init_condition, dims = [2,3])[:]
    N_grp_msm = sum(init_condition, dims = 1)[1, :, :]
    Λ, B = setup_transmission_matrix(ms, ps, N_clique; ingroup=ingroup)

    #Project forwards

    _p = copy([p_trans, R0_other, γ_eff, α_incubation, vac_effectiveness])
    prob = DiscreteProblem((du, u, p, t) -> MonkeypoxUK.f_mpx_vac(du, u, p, t, Λ, B, N_msm, N_grp_msm, N_total),
                        init_condition, 
                        (ts_0, ts_0 + 7 * weeks_to_project),
                        _p)
    mpx_init = init(prob, FunctionMap(), save_everystep=false) #Begins week 1

    #Set up arrays for tracking epidemiological observables
    old_onsets = [0, 0]
    new_onsets = [0, 0]
    old_sus = [sum(u0_msm[1, :, :][:,:],dims = 2)[:]; u0_other[1]]
    new_sus = [sum(u0_msm[1, :, :][:,:],dims = 2)[:]; u0_other[1]]
    wk_num = 1
    detected_cases = zeros(weeks_to_project, 2)
    incidence = zeros(Int64, weeks_to_project, 11)
    not_changed = true
    not_changed2 = true

    #Step through the dynamics
    while wk_num <= weeks_to_project #Step forward a week
        #Change points
        if not_changed && mpx_init.t > chp_t ##1st change point for transmission prob
            not_changed = false
            mpx_init.p[1] = mpx_init.p[1] * (1 - trans_red) #Reduce transmission after the change point
            mpx_init.p[2] = mpx_init.p[2] * (1 - trans_red_other) #Reduce non-sexual transmission after the change point
        end
        if not_changed2 && mpx_init.t > chp_t2 ##2nd change point for transmission 
            not_changed2 = false
            mpx_init.p[1] = mpx_init.p[1] * (1 - trans_red2) #Reduce sexual MSM transmission after the change point
            mpx_init.p[2] = mpx_init.p[2] * (1 - trans_red_other2) #Reduce  other transmission after the change point
        end
        #Step forward a week in time and implement reversion to normal transmission
        # step!(mpx_init, 7)
        # if wk_num >= 19 && mpx_init.p[1] < p_trans  #Reversion starts first week in September
        #     mpx_init.p[1] *= wkly_reversion
        #     mpx_init.p[2] *= wkly_reversion_othr
        # end
        for stp = 1:7
            step!(mpx_init, 1)
            mpx_init.p[1] += (p_trans - p_min)*(sigmoid((mpx_init.t - T₅₀)/κ) - sigmoid((mpx_init.t - 1.0 - T₅₀)/κ))
            mpx_init.p[2] += (R0_other - R_oth_min)*(sigmoid((mpx_init.t - T₅₀)/κ) - sigmoid((mpx_init.t - 1.0 - T₅₀)/κ))
        end

        #Do vaccine uptake
        nv = wkly_vaccinations[wk_num]#Mean number of vaccines deployed
        du_vac = deepcopy(mpx_init.u)
        vac_rate = nv .* du_vac.x[1][1, 3:end, :] / (sum(du_vac.x[1][1, 3:end, :]) .+ 1e-5)
        num_vaccines = map((μ, maxval) -> min(rand(Poisson(μ)), maxval), vac_rate, du_vac.x[1][1, 3:end, :])
        du_vac.x[1][1, 3:end, :] .-= num_vaccines
        du_vac.x[1][8, 3:end, :] .+= num_vaccines
        set_u!(mpx_init, du_vac) #Change the state of the model

        #Calculate actual onsets, actual infections, actual prevelance, generate observed cases and score errors
        new_onsets = [sum(mpx_init.u.x[1][end, :, :]), mpx_init.u.x[2][end]]
        new_sus = [sum(mpx_init.u.x[1][1, :, :][:,:],dims = 2)[:]; mpx_init.u.x[2][1]]
        actual_obs = [rand(BetaBinomial(new_onsets[1] - old_onsets[1], p_detect * M, (1 - p_detect) * M)), rand(BetaBinomial(new_onsets[2] - old_onsets[2], p_detect * M, (1 - p_detect) * M))]
        detected_cases[wk_num, :] .= Float64.(actual_obs)
        incidence[wk_num, :] .= old_sus .- new_sus .- [0;0;sum(num_vaccines,dims = 2)[:];0] #Total infections = reduction in susceptibles - number vaccinated
        if wk_num < size(wkly_cases, 1) # Only compare on weeks 1 --- (end-1)
            L1_rel_err += sum(abs, actual_obs .- wkly_cases[wk_num, :]) / total_cases
        end
        prevalence[wk_num, :] .= [[sum(mpx_init.u.x[1][2:6, n, :]) for n = 1:10]; sum(mpx_init.u.x[2][2:6])]

        #Move time forwards one week
        wk_num += 1
        old_onsets = new_onsets
        old_sus = new_sus
    end

    return L1_rel_err, detected_cases, incidence, prevalence
end