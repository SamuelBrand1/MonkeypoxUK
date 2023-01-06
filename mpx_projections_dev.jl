"""
    function mpx_sim_function_projections(params, constants, interventions, init_condition)

Project forwards from an initial state `init_condition`, with model parameters `params` and vaccine deployment defined by the `NamedTuple` object `interventions`. This model assumes
not reversion in behavioural risk of transmission.        
"""
function mpx_sim_function_projections(params, constants, interventions, init_condition)
    #Get constant data
    N_total,
    N_msm,
    ps,
    ms,
    ingroup,
    ts,
    α_incubation,
    γ_eff,
    epsilon,
    n_cliques,
    wkly_vaccinations,
    vac_effectiveness,
    chp_t2,
    weeks_to_change = constants

    # #Get intervention data
    ts = interventions.ts
    wkly_vaccinations = interventions.wkly_vaccinations
    vac_effectiveness = interventions.vac_effectiveness

    # #Get parameters and make transformations
    α_choose,
    p_detect,
    p_trans,
    R0_other,
    M,
    init_scale,
    chp_t,
    trans_red,
    trans_red_other,
    trans_red2,
    trans_red_other2 = params
    weeks_to_project = length(ts)
    ts_0 = ts[1]
    κ_cng = (weeks_to_change * 7 / 2) / 4.6 # logistic scale for behaviour change to occur over: 4.6 is κ = 1 time to go from 0.01 to 0.5 and 0.5 to 0.99

    # # Generate transmission matrices based on the inital state
    N_clique = sum(init_condition.x[1], dims = [1, 2])[:]
    N_grp_msm = sum(init_condition.x[1], dims = 1)[1, :, :]
    Λ, B = MonkeypoxUK.setup_transmission_matrix(ms, ps, N_clique; ingroup = ingroup)

    #Project forwards

    _p = copy([p_trans, R0_other, γ_eff, α_incubation, vac_effectiveness])
    prob = DiscreteProblem(
        (du, u, p, t) -> MonkeypoxUK.f_mpx_vac(
            du,
            u,
            p,
            t,
            Λ,
            B,
            N_msm,
            N_grp_msm,
            N_total,
            epsilon,
        ),
        init_condition,
        (ts_0, ts_0 + 7 * weeks_to_project),
        _p,
    )
    mpx_init = init(prob, FunctionMap(), save_everystep = false) #Begins week 1

    # #Set up arrays for tracking epidemiological observables

    old_onsets = [sum(init_condition.x[1][end, :, :]), init_condition.x[2][end]]
    new_onsets = [sum(init_condition.x[1][end, :, :]), init_condition.x[2][end]]
    old_sus = [sum(init_condition.x[1][1, :, :][:, :], dims = 2)[:]; init_condition.x[2][1]]
    new_sus = [sum(init_condition.x[1][1, :, :][:, :], dims = 2)[:]; init_condition.x[2][1]]
    wk_num = 1
    detected_cases = zeros(weeks_to_project, 2)
    incidence = zeros(Int64, weeks_to_project, 11)


    # #Step through the dynamics
    while wk_num <= weeks_to_project #Step forward a week

        for day = 1:7
            #Calculate effective transmission rates for each day of transmission
            mpx_init.p[1] =
                mpx_init.t < chp_t2 ?
                p_trans * (1 - trans_red * sigmoid((mpx_init.t - chp_t) / κ_cng)) :
                p_trans *
                (1 - trans_red * sigmoid((mpx_init.t - chp_t) / κ_cng)) *
                (1 - trans_red2)
            mpx_init.p[2] =
                mpx_init.t < chp_t2 ?
                R0_other * (1 - trans_red_other * sigmoid((mpx_init.t - chp_t) / κ_cng)) :
                R0_other *
                (1 - trans_red_other * sigmoid((mpx_init.t - chp_t) / κ_cng)) *
                (1 - trans_red_other2)

            step!(mpx_init, 1)
        end

        #Do vaccine uptake
        nv = wkly_vaccinations[wk_num]#Mean number of vaccines deployed
        du_vac = deepcopy(mpx_init.u)
        vac_rate = nv .* du_vac.x[1][1, 3:end, :] / (sum(du_vac.x[1][1, 3:end, :]) .+ 1e-5)
        num_vaccines = map(
            (μ, maxval) -> min(rand(Poisson(μ)), maxval),
            vac_rate,
            du_vac.x[1][1, 3:end, :],
        )
        du_vac.x[1][1, 3:end, :] .-= num_vaccines
        du_vac.x[1][6, 3:end, :] .+= num_vaccines
        set_u!(mpx_init, du_vac) #Change the state of the model

        #Calculate actual onsets, actual infections, actual prevelance, generate observed cases and score errors
        new_onsets = [sum(mpx_init.u.x[1][end, :, :]), mpx_init.u.x[2][end]]
        new_sus = [sum(mpx_init.u.x[1][1, :, :][:, :], dims = 2)[:]; mpx_init.u.x[2][1]]
        actual_obs = [
            rand(
                BetaBinomial(
                    new_onsets[1] - old_onsets[1],
                    p_detect * M,
                    (1 - p_detect) * M,
                ),
            ),
            rand(
                BetaBinomial(
                    new_onsets[2] - old_onsets[2],
                    p_detect * M,
                    (1 - p_detect) * M,
                ),
            ),
        ]
        detected_cases[wk_num, :] .= Float64.(actual_obs)
        incidence[wk_num, :] .=
            old_sus .- new_sus .- [0; 0; sum(num_vaccines, dims = 2)[:]; 0] #Total infections = reduction in susceptibles - number vaccinated

        #Move time forwards one week
        wk_num += 1
        old_onsets = new_onsets
        old_sus = new_sus
    end

    return (; detected_cases, incidence)
end


"""
    function mpx_sim_function_projections(params, constants, interventions, init_condition, weeks_to_reversion)

Project forwards from an initial state `init_condition`, with model parameters `params` and vaccine deployment defined by the `NamedTuple` object `interventions`. This model assumes
that there is reversion in behavioural risk of transmission starting in the first week of September which occurs over `weeks_to_reversion` weeks.        
"""
function mpx_sim_function_projections(
    params,
    constants,
    interventions,
    init_condition,
    weeks_to_reversion,
)
    #Get constant data
    N_total,
    N_msm,
    ps,
    ms,
    ingroup,
    ts,
    α_incubation,
    γ_eff,
    epsilon,
    n_cliques,
    wkly_vaccinations,
    vac_effectiveness,
    chp_t2,
    weeks_to_change = constants

    # #Get intervention data
    ts = interventions.ts
    wkly_vaccinations = interventions.wkly_vaccinations
    vac_effectiveness = interventions.vac_effectiveness

    # #Get parameters and make transformations
    α_choose,
    p_detect,
    p_trans,
    R0_other,
    M,
    init_scale,
    chp_t,
    trans_red,
    trans_red_other,
    trans_red2,
    trans_red_other2 = params
    weeks_to_project = length(ts)
    ts_0 = ts[1]
    κ_cng = (weeks_to_change * 7 / 2) / 4.6 # logistic scale for behaviour change to occur over: 4.6 is κ = 1 time to go from 0.01 to 0.5 and 0.5 to 0.99

    # Calculate logistic reversion to pre-change
    p_min = p_trans * (1 - trans_red) * (1 - trans_red2)
    R_oth_min = R0_other * (1 - trans_red_other) * (1 - trans_red_other2)
    T₅₀ = (Date(2022, 9, 1) - Date(2021, 12, 31)).value + (weeks_to_reversion * 7 / 2) # 50% return to normal point
    κ_rev = (weeks_to_reversion * 7 / 2) / 4.6 # logistic scale for return to normal: 4.6 is κ = 1 time to go from 0.01 to 0.5 and 0.5 to 0.99

    # # Generate transmission matrices based on the inital state
    N_clique = sum(init_condition.x[1], dims = [1, 2])[:]
    N_grp_msm = sum(init_condition.x[1], dims = 1)[1, :, :]
    Λ, B = MonkeypoxUK.setup_transmission_matrix(ms, ps, N_clique; ingroup = ingroup)

    #Project forwards

    _p = copy([p_trans, R0_other, γ_eff, α_incubation, vac_effectiveness])
    prob = DiscreteProblem(
        (du, u, p, t) -> MonkeypoxUK.f_mpx_vac(
            du,
            u,
            p,
            t,
            Λ,
            B,
            N_msm,
            N_grp_msm,
            N_total,
            epsilon,
        ),
        init_condition,
        (ts_0, ts_0 + 7 * weeks_to_project),
        _p,
    )
    mpx_init = init(prob, FunctionMap(), save_everystep = false) #Begins week 1

    # #Set up arrays for tracking epidemiological observables

    old_onsets = [sum(init_condition.x[1][end, :, :]), init_condition.x[2][end]]
    new_onsets = [sum(init_condition.x[1][end, :, :]), init_condition.x[2][end]]
    old_sus = [sum(init_condition.x[1][1, :, :][:, :], dims = 2)[:]; init_condition.x[2][1]]
    new_sus = [sum(init_condition.x[1][1, :, :][:, :], dims = 2)[:]; init_condition.x[2][1]]
    wk_num = 1
    detected_cases = zeros(weeks_to_project, 2)
    incidence = zeros(Int64, weeks_to_project, 11)


    # #Step through the dynamics
    while wk_num <= weeks_to_project #Step forward a week

        for day = 1:7
            #Calculate effective transmission rates for each day of transmission
            if mpx_init.t < (Date(2022, 9, 1) - Date(2021, 12, 31)).value
                mpx_init.p[1] =
                    mpx_init.t < chp_t2 ?
                    p_trans * (1 - trans_red * sigmoid((mpx_init.t - chp_t) / κ_cng)) :
                    p_trans *
                    (1 - trans_red * sigmoid((mpx_init.t - chp_t) / κ_cng)) *
                    (1 - trans_red2)
                mpx_init.p[2] =
                    mpx_init.t < chp_t2 ?
                    R0_other *
                    (1 - trans_red_other * sigmoid((mpx_init.t - chp_t) / κ_cng)) :
                    R0_other *
                    (1 - trans_red_other * sigmoid((mpx_init.t - chp_t) / κ_cng)) *
                    (1 - trans_red_other2)
            else
                mpx_init.p[1] =
                    p_min + (p_trans - p_min) * sigmoid((mpx_init.t - T₅₀) / κ_rev)
                mpx_init.p[2] =
                    R_oth_min + (R0_other - R_oth_min) * sigmoid((mpx_init.t - T₅₀) / κ_rev)
            end
            step!(mpx_init, 1)
        end

        #Do vaccine uptake
        nv = wkly_vaccinations[wk_num]#Mean number of vaccines deployed
        du_vac = deepcopy(mpx_init.u)
        vac_rate = nv .* du_vac.x[1][1, 3:end, :] / (sum(du_vac.x[1][1, 3:end, :]) .+ 1e-5)
        num_vaccines = map(
            (μ, maxval) -> min(rand(Poisson(μ)), maxval),
            vac_rate,
            du_vac.x[1][1, 3:end, :],
        )
        du_vac.x[1][1, 3:end, :] .-= num_vaccines
        du_vac.x[1][6, 3:end, :] .+= num_vaccines
        set_u!(mpx_init, du_vac) #Change the state of the model

        #Calculate actual onsets, actual infections, actual prevelance, generate observed cases and score errors
        new_onsets = [sum(mpx_init.u.x[1][end, :, :]), mpx_init.u.x[2][end]]
        new_sus = [sum(mpx_init.u.x[1][1, :, :][:, :], dims = 2)[:]; mpx_init.u.x[2][1]]
        actual_obs = [
            rand(
                BetaBinomial(
                    new_onsets[1] - old_onsets[1],
                    p_detect * M,
                    (1 - p_detect) * M,
                ),
            ),
            rand(
                BetaBinomial(
                    new_onsets[2] - old_onsets[2],
                    p_detect * M,
                    (1 - p_detect) * M,
                ),
            ),
        ]
        detected_cases[wk_num, :] .= Float64.(actual_obs)
        incidence[wk_num, :] .=
            old_sus .- new_sus .- [0; 0; sum(num_vaccines, dims = 2)[:]; 0] #Total infections = reduction in susceptibles - number vaccinated

        #Move time forwards one week
        wk_num += 1
        old_onsets = new_onsets
        old_sus = new_sus
    end

    return (; detected_cases, incidence)
end


"""
    function reproductive_ratios(
        params,
        constants,
        interventions,
        weeks_to_reversion;
        av_cnt_rate=mean(mean_daily_cnts),
        reversion_time=(Date(2022, 9, 1) - Date(2021, 12, 31)).value
    )

Calculate the R₀ and (instaneous) R_t for GBMSM and non-GBMSM, with a 1% -> 99% reversion to normal occuring over `days_reversion`.
"""
function reproductive_ratios(
    params,
    constants,
    weeks_to_reversion,
    ts;
    av_cnt_rate = mean(mean_daily_cnts),
    reversion_time = (Date(2022, 9, 1) - Date(2021, 12, 31)).value,
)
    #Get constant data
    N_total,
    N_msm,
    ps,
    ms,
    ingroup,
    ___,
    α_incubation,
    γ_eff,
    epsilon,
    n_cliques,
    _,
    __,
    chp_t2,
    weeks_to_change = constants

    # #Get parameters and make transformations
    α_choose,
    p_detect,
    p_trans,
    R0_other,
    M,
    init_scale,
    chp_t,
    trans_red,
    trans_red_other,
    trans_red2,
    trans_red_other2 = params
    κ_cng = (weeks_to_change * 7 / 2) / 4.6 # logistic scale for behaviour change to occur over: 4.6 is κ = 1 time to go from 0.01 to 0.5 and 0.5 to 0.99
    mean_inf_period = (epsilon / (1 - exp(-α_incubation))) + (1 / (1 - exp(-γ_eff)))

    # Calculate logistic reversion to pre-change
    p_min = p_trans * (1 - trans_red) * (1 - trans_red2)
    R_oth_min = R0_other * (1 - trans_red_other) * (1 - trans_red_other2)
    T₅₀ = reversion_time + (weeks_to_reversion * 7 / 2) # 50% return to normal point
    κ_rev = (weeks_to_reversion * 7 / 2) / 4.6 # logistic scale for return to normal: 4.6 is κ = 1 time to go from 0.01 to 0.5 and 0.5 to 0.99

    # Calculate 
    R₀_gbmsm = Vector{Float64}(undef, length(ts))
    R₀_ngbmsm = Vector{Float64}(undef, length(ts))

    for (i, t) in enumerate(ts)
        if t < reversion_time
            R₀_gbmsm[i] =
                t < chp_t2 ? p_trans * (1 - trans_red * sigmoid((t - chp_t) / κ_cng)) :
                p_trans * (1 - trans_red * sigmoid((t - chp_t) / κ_cng)) * (1 - trans_red2)
            R₀_ngbmsm[i] =
                t < chp_t2 ?
                R0_other * (1 - trans_red_other * sigmoid((t - chp_t) / κ_cng)) :
                R0_other *
                (1 - trans_red_other * sigmoid((t - chp_t) / κ_cng)) *
                (1 - trans_red_other2)
            R₀_gbmsm[i] *= mean_inf_period * av_cnt_rate
        else
            R₀_gbmsm[i] = p_min + (p_trans - p_min) * sigmoid((t - T₅₀) / κ_rev)
            R₀_ngbmsm[i] = R_oth_min + (R0_other - R_oth_min) * sigmoid((t - T₅₀) / κ_rev)
            R₀_gbmsm[i] *= mean_inf_period * av_cnt_rate
        end
    end

    return (; R₀_gbmsm, R₀_ngbmsm)
end


"""
    function reproductive_ratios(
        params,
        constants,
        interventions,
        weeks_to_reversion;
        av_cnt_rate=mean(mean_daily_cnts),
        reversion_time=(Date(2022, 9, 1) - Date(2021, 12, 31)).value
    )

Calculate the R₀ and (instaneous) R_t for GBMSM and non-GBMSM, with a 1% -> 99% reversion to normal occuring over `days_reversion`.
"""
function reproductive_ratios(
    params,
    constants,
    ts;
    av_cnt_rate = mean(mean_daily_cnts),
    reversion_time = (Date(2022, 9, 1) - Date(2021, 12, 31)).value,
)
    #Get constant data
    N_total,
    N_msm,
    ps,
    ms,
    ingroup,
    ___,
    α_incubation,
    γ_eff,
    epsilon,
    n_cliques,
    _,
    __,
    chp_t2,
    weeks_to_change = constants

    # #Get parameters and make transformations
    α_choose,
    p_detect,
    p_trans,
    R0_other,
    M,
    init_scale,
    chp_t,
    trans_red,
    trans_red_other,
    trans_red2,
    trans_red_other2 = params
    κ_cng = (weeks_to_change * 7 / 2) / 4.6 # logistic scale for behaviour change to occur over: 4.6 is κ = 1 time to go from 0.01 to 0.5 and 0.5 to 0.99
    mean_inf_period = (epsilon / (1 - exp(-α_incubation))) + (1 / (1 - exp(-γ_eff)))


    # Calculate 
    R₀_gbmsm = Vector{Float64}(undef, length(ts))
    R₀_ngbmsm = Vector{Float64}(undef, length(ts))

    for (i, t) in enumerate(ts)
        R₀_gbmsm[i] =
            t < chp_t2 ? p_trans * (1 - trans_red * sigmoid((t - chp_t) / κ_cng)) :
            p_trans * (1 - trans_red * sigmoid((t - chp_t) / κ_cng)) * (1 - trans_red2)
        R₀_ngbmsm[i] =
            t < chp_t2 ? R0_other * (1 - trans_red_other * sigmoid((t - chp_t) / κ_cng)) :
            R0_other *
            (1 - trans_red_other * sigmoid((t - chp_t) / κ_cng)) *
            (1 - trans_red_other2)
        R₀_gbmsm[i] *= mean_inf_period * av_cnt_rate
    end

    return (; R₀_gbmsm, R₀_ngbmsm)
end
