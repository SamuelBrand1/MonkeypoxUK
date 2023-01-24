"""
    function setup_initial_state(N_pop, N_msm, α_choose, p_detect, α_incubation_eff, ps, init_scale; n_states=8, n_cliques=50)

Setup an initial state for the MPX model given overall population size `N_pop`, overall MSM population size `N_msm`, metapopulation size dispersion parameter `α_choose`,
    mean weekly probability of detection `p_detect`, incubation rate `α_incubation_eff`, proportion MSM in each sexual activity group `ps`, and initial scale of number of infected `init_scale`. Output 
    initial MSM population array `u0_msm`, initial non-MSM array `u0_other`, metapopulation sizes, total population in each metapopulation/sexual activity group.
"""
function setup_initial_state(
    N_pop,
    N_msm,
    α_choose,
    p_detect,
    α_incubation_eff,
    ps,
    init_scale;
    n_states = 7,
    n_cliques = 50,
    n_infected_states = 3,
)
    u0_msm = zeros(Int64, n_states, length(ps), n_cliques)
    u0_other = zeros(Int64, n_states)
    N_clique = rand(DirichletMultinomial(N_msm, α_choose * ones(n_cliques)))
    for k = 1:n_cliques
        u0_msm[1, :, k] .= rand(Multinomial(N_clique[k], ps))
    end
    #Add infecteds so that expected detections on week 1 are 1
    choose_clique = rand(Categorical(normalize(N_clique, 1)))
    av_infs = n_infected_states * init_scale / (p_detect * (1 - exp(-α_incubation_eff))) #Set av number of infecteds in each of `n_infected_states` categories of incubation and infectious rescaled by daily probability of detection if infectious
    u0_msm[2:(n_infected_states+1), :, choose_clique] .= map(
        μ -> rand(Poisson(μ)),
        fill(av_infs / (n_infected_states * length(ps)), n_infected_states, length(ps)),
    )
    u0_msm = u0_msm[:, :, N_clique.>0] #Reduce group size
    #Set up non MSM population
    u0_other[1] = N_pop - N_msm
    N_grp_msm = u0_msm[1, :, :]
    return u0_msm, u0_other, N_clique[N_clique.>0], N_grp_msm
end

"""
    function setup_transmission_matrix(ms, ps, N_clique; ingroup=0.99)

Setup the two matrices used in calculating daily force of infection due to infectious MSM individuals.        
"""
function setup_transmission_matrix(ms, ps, N_clique; ingroup = 0.99)
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


"""
    function f_mpx_vac(du, u, p, t, Λ, B, N_msm, N_grp_msm, N_total)

Daily stochastic dynamics of the MPX model. NB: the state object `u` is an `ArrayPartition` object with 
        `u.x[1]` being the state of the MSM population and `u.x[2]` being the state of the non-MSM population.

"""
function f_mpx_vac(du, u, p, t, Λ, B, N_msm, N_grp_msm, N_total, ϵ)
    # Parameters
    p_trans, R0_other, γ_eff, α_incubation, vac_eff = p

    # States
    S = @view u.x[1][1, :, :]
    E = @view u.x[1][2, :, :] #Latent AND pre-symptomatic infectious
    P = @view u.x[1][3, :, :] #Just pre-symptomatic infectious
    I = @view u.x[1][4, :, :]
    R = @view u.x[1][5, :, :]
    V = @view u.x[1][6, :, :]
    C = @view u.x[1][7, :, :]

    S_other = u.x[2][1]
    E_other = u.x[2][2] #Latent AND pre-symptomatic infectious
    P_other = u.x[2][3] #Just pre-symptomatic infectious
    I_other = u.x[2][4]
    R_other = u.x[2][5]

    # Force of infection
    total_I = I_other + sum(I) + ϵ * (P_other + sum(P))
    λ = (p_trans .* (Λ * (I .+ ϵ .* P) * B)) ./ (N_grp_msm .+ 1e-5)
    λ_other = (1 - exp(-γ_eff)) * R0_other * total_I / N_total

    # Number of events

    num_infs = map((n, p) -> rand(Binomial(n, p)), S, 1 .- exp.(-(λ .+ λ_other))) # infections among MSM
    num_infs_vac =
        map((n, p) -> rand(Binomial(n, p)), V, 1 .- exp.(-(1 - vac_eff) * (λ .+ λ_other))) # infections among MSM with vaccine
    num_incs1 = map(n -> rand(Binomial(n, 1 - exp(-α_incubation))), E) # incubation among MSM
    num_incs2 = map(n -> rand(Binomial(n, 1 - exp(-α_incubation))), P) # incubation among MSM
    num_recs = map(n -> rand(Binomial(n, 1 - exp(-γ_eff))), I)#recovery among MSM
    num_infs_other = rand(Binomial(S_other, 1 - exp(-λ_other)))#infections among non MSM
    num_incs_other1 = map(n -> rand(Binomial(n, 1 - exp(-α_incubation))), E_other)#incubation among non MSM
    num_incs_other2 = map(n -> rand(Binomial(n, 1 - exp(-α_incubation))), P_other)#incubation among non MSM

    num_recs_other = rand(Binomial(I_other, 1 - exp(-γ_eff)))#recovery among non MSM

    # Create change
    du.x[1] .= u.x[1]
    du.x[2] .= u.x[2]
    #infections
    du.x[1][1, :, :] .-= num_infs
    du.x[1][6, :, :] .-= num_infs_vac
    du.x[1][2, :, :] .+= num_infs .+ num_infs_vac
    du.x[2][1] -= num_infs_other
    du.x[2][2] += num_infs_other
    #incubations
    du.x[1][2, :, :] .-= num_incs1
    du.x[1][3, :, :] .+= num_incs1
    du.x[1][3, :, :] .-= num_incs2
    du.x[1][4, :, :] .+= num_incs2
    du.x[2][2] -= num_incs_other1
    du.x[2][3] += num_incs_other1
    du.x[2][3] -= num_incs_other2
    du.x[2][4] += num_incs_other2
    du.x[1][7, :, :] .+= num_incs2 #cumulative onsets - MSM
    du.x[2][7] += num_incs_other2 #cumulative onsets - non-MSM
    #recoveries
    du.x[1][4, :, :] .-= num_recs
    du.x[1][5, :, :] .+= num_recs
    du.x[2][4] -= num_recs_other
    du.x[2][5] += num_recs_other

    return nothing
end

"""
    function mpx_sim_function_chp(params, constants, wkly_cases)

Simulation function for the MPX transmission model with change points. Outputs the relative L1 error for ABC inference, 
    and detected cases (for saving).        
"""
function mpx_sim_function_chp(params, constants, wkly_cases)
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

    #Get parameters and make parameter transformations transformations
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


    vac_effectiveness = rand(Uniform(0.7, 0.85))
    κ = (weeks_to_change * 7 / 2) / 4.6 # logistic scale for behaviour change to occur over: 4.6 is κ = 1 time to go from 0.01 to 0.5 and 0.5 to 0.99

    #Generate random population structure
    u0_msm, u0_other, N_clique, N_grp_msm = setup_initial_state(
        N_total,
        N_msm,
        α_choose,
        p_detect,
        α_incubation,
        ps,
        init_scale;
        n_cliques = n_cliques,
    )
    Λ, B = setup_transmission_matrix(ms, ps, N_clique; ingroup = ingroup)

    #Simulate and track error
    L1_rel_err = 0.0
    total_cases = sum(wkly_cases[1:(end-1), :])

    #Initialise simulation
    u_mpx = ArrayPartition(u0_msm, u0_other)
    start_state = deepcopy(u_mpx)

    prob = DiscreteProblem(
        (du, u, p, t) ->
            f_mpx_vac(du, u, p, t, Λ, B, N_msm, N_grp_msm, N_total, epsilon), #Simulation update function
        u_mpx, #Initial state
        (ts[1] - 7, ts[1] - 7 + 7 * size(wkly_cases, 1)), #Start week before detection
        [p_trans, R0_other, γ_eff, α_incubation, vac_effectiveness],
    ) #Basic simulation parameters
    mpx_init = init(prob, FunctionMap(), save_everystep = false) #Begins week 1

    #Initialise arrays tracking observables and generated quantities of the simulation
    old_onsets = [0, 0]
    new_onsets = [0, 0]
    old_sus = [sum(u0_msm[1, :, :][:, :], dims = 2)[:]; u0_other[1]]
    new_sus = [sum(u0_msm[1, :, :][:, :], dims = 2)[:]; u0_other[1]]
    wk_num = 1
    detected_cases = zeros(size(wkly_cases))
    onsets = zeros(size(wkly_cases))
    incidence = zeros(Int64, size(wkly_cases, 1), length(ps) + 1)
    susceptibility = zeros(size(wkly_cases, 1), length(ps) + 1)
    state_pre_vaccine = similar(u_mpx)
    state_sept = similar(u_mpx)
    wk_vac = findfirst(wkly_vaccinations .> 0)
    wk_sept = findfirst(ts .> 244)

    #Dynamics
    while wk_num <= size(wkly_cases, 1)
        #Save states at important moments
        if wk_num == wk_vac
            state_pre_vaccine = deepcopy(mpx_init.u)
        end
        if wk_num == wk_sept
            state_sept = deepcopy(mpx_init.u)
        end

        #Step dynamics forward a week
        for day = 1:7
            #Calculate effective transmission rates for each day of transmission
            mpx_init.p[1] =
                mpx_init.t < chp_t2 ?
                p_trans * (1 - trans_red * sigmoid((mpx_init.t - chp_t) / κ)) :
                p_trans *
                (1 - trans_red * sigmoid((mpx_init.t - chp_t) / κ)) *
                (1 - trans_red2)
            mpx_init.p[2] =
                mpx_init.t < chp_t2 ?
                R0_other * (1 - trans_red_other * sigmoid((mpx_init.t - chp_t) / κ)) :
                R0_other *
                (1 - trans_red_other * sigmoid((mpx_init.t - chp_t) / κ)) *
                (1 - trans_red_other2)
            step!(mpx_init, 1) # Dynamics for day
        end

        #Do vaccine uptake
        nv = wkly_vaccinations[wk_num]#Mean number of vaccines deployed
        du_vac = deepcopy(mpx_init.u)
        vac_rate = nv .* du_vac.x[1][1, 3:end, :] / (sum(du_vac.x[1][1:3, 3:end, :]) .+ 1e-5)
        num_vaccines = map(
            (μ, maxval) -> min(rand(Poisson(μ)), maxval),
            vac_rate,
            du_vac.x[1][1, 3:end, :],
        )
        du_vac.x[1][1, 3:end, :] .-= num_vaccines
        du_vac.x[1][6, 3:end, :] .+= num_vaccines
        set_u!(mpx_init, du_vac) #Change the state of the model

        #Calculate actual onsets, generate observed cases and score errors        
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
        detected_cases[wk_num, :] .= actual_obs #lag 1 week
        onsets[wk_num, :] .= new_onsets #lag 1 week
        incidence[wk_num, :] .=
            old_sus .- new_sus .- [0; 0; sum(num_vaccines, dims = 2)[:]; 0] #Total infections = reduction in susceptibles - number vaccinated
        susceptibility[wk_num, :] .=
            (
                new_sus .+
                (1 - vac_effectiveness) .*
                [sum(mpx_init.u.x[1][6, :, :][:, :], dims = 2)[:]; 0]
            ) ./ [N_msm .* ps; N_total - N_msm]

        if wk_num < size(wkly_cases, 1)  # Leave last week out due to right censoring issues 
            L1_rel_err += sum(abs, actual_obs .- wkly_cases[wk_num, :]) / total_cases #lag 1 week
        end

        wk_num += 1
        old_onsets = new_onsets
        old_sus = new_sus
    end

    end_state = mpx_init.u #For doing projections

    return L1_rel_err,
    (;
        detected_cases,
        onsets,
        incidence,
        susceptibility,
        vac_effectiveness,
        state_pre_vaccine,
        state_sept,
        end_state,
        start_state,
    )
end

"""
    function mpx_sim_function_chp_uniform_vac(params, constants, wkly_cases)

Simulation function for the MPX transmission model with change points. Outputs the relative L1 error for ABC inference, 
    and detected cases (for saving). This version has uniform vaccination across all GBMSM groups.        
"""
function mpx_sim_function_chp_uniform_vac(params, constants, wkly_cases)
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

    #Get parameters and make parameter transformations transformations
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


    vac_effectiveness = rand(Uniform(0.7, 0.85))
    κ = (weeks_to_change * 7 / 2) / 4.6 # logistic scale for behaviour change to occur over: 4.6 is κ = 1 time to go from 0.01 to 0.5 and 0.5 to 0.99

    #Generate random population structure
    u0_msm, u0_other, N_clique, N_grp_msm = setup_initial_state(
        N_total,
        N_msm,
        α_choose,
        p_detect,
        α_incubation,
        ps,
        init_scale;
        n_cliques = n_cliques,
    )
    Λ, B = setup_transmission_matrix(ms, ps, N_clique; ingroup = ingroup)

    #Simulate and track error
    L1_rel_err = 0.0
    total_cases = sum(wkly_cases[1:(end-1), :])

    #Initialise simulation
    u_mpx = ArrayPartition(u0_msm, u0_other)
    start_state = deepcopy(u_mpx)

    prob = DiscreteProblem(
        (du, u, p, t) ->
            f_mpx_vac(du, u, p, t, Λ, B, N_msm, N_grp_msm, N_total, epsilon), #Simulation update function
        u_mpx, #Initial state
        (ts[1] - 7, ts[1] - 7 + 7 * size(wkly_cases, 1)), #Start week before detection
        [p_trans, R0_other, γ_eff, α_incubation, vac_effectiveness],
    ) #Basic simulation parameters
    mpx_init = init(prob, FunctionMap(), save_everystep = false) #Begins week 1

    #Initialise arrays tracking observables and generated quantities of the simulation
    old_onsets = [0, 0]
    new_onsets = [0, 0]
    old_sus = [sum(u0_msm[1, :, :][:, :], dims = 2)[:]; u0_other[1]]
    new_sus = [sum(u0_msm[1, :, :][:, :], dims = 2)[:]; u0_other[1]]
    wk_num = 1
    detected_cases = zeros(size(wkly_cases))
    onsets = zeros(size(wkly_cases))
    incidence = zeros(Int64, size(wkly_cases, 1), length(ps) + 1)
    susceptibility = zeros(size(wkly_cases, 1), length(ps) + 1)
    state_pre_vaccine = similar(u_mpx)
    state_sept = similar(u_mpx)
    wk_vac = findfirst(wkly_vaccinations .> 0)
    wk_sept = findfirst(ts .> 244)

    #Dynamics
    while wk_num <= size(wkly_cases, 1)
        #Save states at important moments
        if wk_num == wk_vac
            state_pre_vaccine = deepcopy(mpx_init.u)
        end
        if wk_num == wk_sept
            state_sept = deepcopy(mpx_init.u)
        end

        #Step dynamics forward a week
        for day = 1:7
            #Calculate effective transmission rates for each day of transmission
            mpx_init.p[1] =
                mpx_init.t < chp_t2 ?
                p_trans * (1 - trans_red * sigmoid((mpx_init.t - chp_t) / κ)) :
                p_trans *
                (1 - trans_red * sigmoid((mpx_init.t - chp_t) / κ)) *
                (1 - trans_red2)
            mpx_init.p[2] =
                mpx_init.t < chp_t2 ?
                R0_other * (1 - trans_red_other * sigmoid((mpx_init.t - chp_t) / κ)) :
                R0_other *
                (1 - trans_red_other * sigmoid((mpx_init.t - chp_t) / κ)) *
                (1 - trans_red_other2)
            step!(mpx_init, 1) # Dynamics for day
        end

        #Do vaccine uptake
        nv = wkly_vaccinations[wk_num] #Mean number of vaccines deployed
        du_vac = deepcopy(mpx_init.u)
        vac_rate = nv .* du_vac.x[1][1, 1:end, :] / (sum(du_vac.x[1][1:3, 1:end, :]) .+ 1e-5)
        num_vaccines = map(
            (μ, maxval) -> min(rand(Poisson(μ)), maxval),
            vac_rate,
            du_vac.x[1][1, 1:end, :],
        )
        du_vac.x[1][1, 1:end, :] .-= num_vaccines
        du_vac.x[1][6, 1:end, :] .+= num_vaccines
        set_u!(mpx_init, du_vac) #Change the state of the model

        #Calculate actual onsets, generate observed cases and score errors        
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
        detected_cases[wk_num, :] .= actual_obs #lag 1 week
        onsets[wk_num, :] .= new_onsets #lag 1 week
        incidence[wk_num, :] .=
            old_sus .- new_sus .- [0; 0; sum(num_vaccines, dims = 2)[:]; 0] #Total infections = reduction in susceptibles - number vaccinated
        susceptibility[wk_num, :] .=
            (
                new_sus .+
                (1 - vac_effectiveness) .*
                [sum(mpx_init.u.x[1][6, :, :][:, :], dims = 2)[:]; 0]
            ) ./ [N_msm .* ps; N_total - N_msm]

        if wk_num < size(wkly_cases, 1)  # Leave last week out due to right censoring issues 
            L1_rel_err += sum(abs, actual_obs .- wkly_cases[wk_num, :]) / total_cases #lag 1 week
        end

        wk_num += 1
        old_onsets = new_onsets
        old_sus = new_sus
    end

    end_state = mpx_init.u #For doing projections

    return L1_rel_err,
    (;
        detected_cases,
        onsets,
        incidence,
        susceptibility,
        vac_effectiveness,
        state_pre_vaccine,
        state_sept,
        end_state,
        start_state,
    )
end


"""
    function mpx_sim_function_mdl_selection(params, constants, wkly_cases)

Simulation function for the MPX transmission model with change points. Outputs the relative L1 error for ABC inference, 
    and detected cases (for saving).        
"""
function mpx_sim_function_mdl_selection(params, constants, wkly_cases)
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

    #Get parameters and make parameter transformations transformations
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
    trans_red_other2,
    α_choose_nb,
    p_detect_nb,
    p_trans_nb,
    R0_other_nb,
    init_scale_nb,
    prob_mdl_choice = params
    mdl_choice = rand(Bernoulli(prob_mdl_choice))
    #Choose model, if true then its the no behaviour change model
    if mdl_choice
        α_choose = α_choose_nb
        p_detect = p_detect_nb
        p_trans = p_trans_nb
        R0_other = R0_other_nb
        init_scale = init_scale_nb
        trans_red = 0.0
        trans_red_other = 0.0
        trans_red2 = 0.0
        trans_red_other2 = 0.0
    end

    vac_effectiveness = rand(Uniform(0.7, 0.85))
    κ = (weeks_to_change * 7 / 2) / 4.6 # logistic scale for behaviour change to occur over: 4.6 is κ = 1 time to go from 0.01 to 0.5 and 0.5 to 0.99
    # trans_red2 = scale_trans_red2
    # trans_red_other2 = scale_red_other2

    #Generate random population structure
    u0_msm, u0_other, N_clique, N_grp_msm = setup_initial_state(
        N_total,
        N_msm,
        α_choose,
        p_detect,
        α_incubation,
        ps,
        init_scale;
        n_cliques = n_cliques,
    )
    Λ, B = setup_transmission_matrix(ms, ps, N_clique; ingroup = ingroup)

    #Simulate and track error
    L1_rel_err = 0.0
    total_cases = sum(wkly_cases[1:(end-1), :])

    #Initialise simulation
    u_mpx = ArrayPartition(u0_msm, u0_other)
    prob = DiscreteProblem(
        (du, u, p, t) ->
            f_mpx_vac(du, u, p, t, Λ, B, N_msm, N_grp_msm, N_total, epsilon), #Simulation update function
        u_mpx, #Initial state
        (ts[1] - 7, ts[1] - 7 + 7 * size(wkly_cases, 1)), #Start week before detection
        [p_trans, R0_other, γ_eff, α_incubation, vac_effectiveness],
    ) #Basic simulation parameters
    mpx_init = init(prob, FunctionMap(), save_everystep = false) #Begins week 1

    #Initialise arrays tracking observables and generated quantities of the simulation
    old_onsets = [0, 0]
    new_onsets = [0, 0]
    old_sus = [sum(u0_msm[1, :, :][:, :], dims = 2)[:]; u0_other[1]]
    new_sus = [sum(u0_msm[1, :, :][:, :], dims = 2)[:]; u0_other[1]]
    wk_num = 1
    detected_cases = zeros(size(wkly_cases))
    onsets = zeros(size(wkly_cases))
    incidence = zeros(Int64, size(wkly_cases, 1), length(ps) + 1)
    susceptibility = zeros(size(wkly_cases, 1), length(ps) + 1)
    state_pre_vaccine = similar(u_mpx)
    state_sept = similar(u_mpx)
    wk_vac = findfirst(wkly_vaccinations .> 0)
    wk_sept = findfirst(ts .> 244)

    #Dynamics
    while wk_num <= size(wkly_cases, 1)
        #Save states at important moments
        if wk_num == wk_vac
            state_pre_vaccine = deepcopy(mpx_init.u)
        end
        if wk_num == wk_sept
            state_sept = deepcopy(mpx_init.u)
        end

        #Step dynamics forward a week
        for day = 1:7
            #Calculate effective transmission rates for each day of transmission
            mpx_init.p[1] =
                mpx_init.t < chp_t2 ?
                p_trans * (1 - trans_red * sigmoid((mpx_init.t - chp_t) / κ)) :
                p_trans *
                (1 - trans_red * sigmoid((mpx_init.t - chp_t) / κ)) *
                (1 - trans_red2)
            mpx_init.p[2] =
                mpx_init.t < chp_t2 ?
                R0_other * (1 - trans_red_other * sigmoid((mpx_init.t - chp_t) / κ)) :
                R0_other *
                (1 - trans_red_other * sigmoid((mpx_init.t - chp_t) / κ)) *
                (1 - trans_red_other2)
            step!(mpx_init, 1) # Dynamics for day
        end

        #Do vaccine uptake
        nv = wkly_vaccinations[wk_num]#Mean number of vaccines deployed
        du_vac = deepcopy(mpx_init.u)
        vac_rate = nv .* du_vac.x[1][1, 3:end, :] / (sum(du_vac.x[1][1:3, 3:end, :]) .+ 1e-5)
        num_vaccines = map(
            (μ, maxval) -> min(rand(Poisson(μ)), maxval),
            vac_rate,
            du_vac.x[1][1, 3:end, :],
        )
        du_vac.x[1][1, 3:end, :] .-= num_vaccines
        du_vac.x[1][6, 3:end, :] .+= num_vaccines
        set_u!(mpx_init, du_vac) #Change the state of the model

        #Calculate actual onsets, generate observed cases and score errors        
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
        detected_cases[wk_num, :] .= actual_obs #lag 1 week
        onsets[wk_num, :] .= new_onsets #lag 1 week
        incidence[wk_num, :] .=
            old_sus .- new_sus .- [0; 0; sum(num_vaccines, dims = 2)[:]; 0] #Total infections = reduction in susceptibles - number vaccinated
        susceptibility[wk_num, :] .=
            (
                new_sus .+
                (1 - vac_effectiveness) .*
                [sum(mpx_init.u.x[1][6, :, :][:, :], dims = 2)[:]; 0]
            ) ./ [N_msm .* ps; N_total - N_msm]

        if wk_num < size(wkly_cases, 1)  # Leave last week out due to right censoring issues 
            L1_rel_err += sum(abs, actual_obs .- wkly_cases[wk_num, :]) / total_cases #lag 1 week
        end

        wk_num += 1
        old_onsets = new_onsets
        old_sus = new_sus
    end

    end_state = mpx_init.u #For doing projections

    return L1_rel_err,
    (;
        detected_cases,
        onsets,
        incidence,
        susceptibility,
        vac_effectiveness,
        state_pre_vaccine,
        state_sept,
        end_state,
        mdl_choice,
    )
end


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
    susceptibility = zeros(weeks_to_project, length(ps) + 1)

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
        vac_rate = nv .* du_vac.x[1][1, 3:end, :] / (sum(du_vac.x[1][1:3, 3:end, :]) .+ 1e-5)
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

        susceptibility[wk_num, :] .=
            (
                new_sus .+
                (1 - vac_effectiveness) .*
                [sum(mpx_init.u.x[1][6, :, :][:, :], dims = 2)[:]; 0]
            ) ./ [N_msm .* ps; N_total - N_msm]

        #Move time forwards one week
        wk_num += 1
        old_onsets = new_onsets
        old_sus = new_sus
    end

    return (; detected_cases, incidence, susceptibility)
end

"""
    function mpx_sim_function_projections_uniform_vaccination(params, constants, interventions, init_condition)

Project forwards from an initial state `init_condition`, with model parameters `params` and vaccine deployment defined by the `NamedTuple` object `interventions`. This model assumes
not reversion in behavioural risk of transmission. Vaccine deployment is assumed to be uniform across all risk groups.        
"""
function mpx_sim_function_projections_uniform_vaccination(params, constants, interventions, init_condition)
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
        vac_rate = nv .* du_vac.x[1][1, 1:end, :] / (sum(du_vac.x[1][1:3, 1:end, :]) .+ 1e-5)
        num_vaccines = map(
            (μ, maxval) -> min(rand(Poisson(μ)), maxval),
            vac_rate,
            du_vac.x[1][1, 1:end, :],
        )
        du_vac.x[1][1, 1:end, :] .-= num_vaccines
        du_vac.x[1][6, 1:end, :] .+= num_vaccines
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

    #Get intervention data
    ts = interventions.ts
    wkly_vaccinations = interventions.wkly_vaccinations
    vac_effectiveness = interventions.vac_effectiveness

    #Get parameters and make transformations
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
    susceptibility = zeros(weeks_to_project, length(ps) + 1)


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
        vac_rate = nv .* du_vac.x[1][1, 3:end, :] / (sum(du_vac.x[1][1:3, 3:end, :]) .+ 1e-5)
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
        susceptibility[wk_num, :] .=
            (
                new_sus .+
                (1 - vac_effectiveness) .*
                [sum(mpx_init.u.x[1][6, :, :][:, :], dims = 2)[:]; 0]
            ) ./ [N_msm .* ps; N_total - N_msm]

        #Move time forwards one week
        wk_num += 1
        old_onsets = new_onsets
        old_sus = new_sus
    end

    return (; detected_cases, incidence, susceptibility)
end

"""
    function mpx_sim_function_projections_uniform_vaccination(params, constants, interventions, init_condition, weeks_to_reversion)

Project forwards from an initial state `init_condition`, with model parameters `params` and vaccine deployment defined by the `NamedTuple` object `interventions`. This model assumes
that there is reversion in behavioural risk of transmission starting in the first week of September which occurs over `weeks_to_reversion` weeks. Vaccination is applied uniformly.        
"""
function mpx_sim_function_projections_uniform_vaccination(
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

    #Get intervention data
    ts = interventions.ts
    wkly_vaccinations = interventions.wkly_vaccinations
    vac_effectiveness = interventions.vac_effectiveness

    #Get parameters and make transformations
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
        vac_rate = nv .* du_vac.x[1][1, 1:end, :] / (sum(du_vac.x[1][1:3, 1:end, :]) .+ 1e-5)
        num_vaccines = map(
            (μ, maxval) -> min(rand(Poisson(μ)), maxval),
            vac_rate,
            du_vac.x[1][1, 1:end, :],
        )
        du_vac.x[1][1, 1:end, :] .-= num_vaccines
        du_vac.x[1][6, 1:end, :] .+= num_vaccines
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
            old_sus .- new_sus .- [sum(num_vaccines, dims = 2)[:]; 0] #Total infections = reduction in susceptibles - number vaccinated

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
    av_cnt_rate = 1.978375735058361,
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
    av_cnt_rate = 1.978375735058361,
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
