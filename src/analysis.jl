"""
    function generate_forecast_projection(param_draws, wks, mpxv_wkly, constants)

Generate the forecast projection from posterior draws of the SMC-ABC algoritm.        
"""
function generate_forecast_projection(param_draws, wks, mpxv_wkly, constants)
    ## Public health emergency effect forecasts
    long_wks = [wks; [wks[end] + Day(7 * k) for k = 1:12]]
    long_mpxv_wkly = [mpxv_wkly; zeros(12, 2)]

    ##Generate main scenarios
    chp_t2 = (Date(2022, 7, 23) - Date(2021, 12, 31)).value #Announcement of Public health emergency
    inf_duration_red = 0.0
    wkly_vaccinations = constants[end-2]

    interventions_ensemble = [
        (
            trans_red2 = θ[9] * θ[11],
            vac_effectiveness = rand(Uniform(0.7, 0.85)),
            trans_red_other2 = θ[10] * θ[12],
            wkly_vaccinations,
            chp_t2,
            inf_duration_red,
        ) for θ in param_draws
    ]


    #Simulate
    preds_and_incidence_interventions = map(
        (θ, intervention) ->
            mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention)[2:4],
        param_draws,
        interventions_ensemble,
    )

    return preds_and_incidence_interventions
end


"""
    function generate_scenario_projections(param_draws, wks, mpxv_wkly, constants)

Generate an ensemble of projections for each scenario from posterior draws of the SMC-ABC algoritm.        
"""
function generate_scenario_projections(param_draws, wks, mpxv_wkly, constants)
    ## Public health emergency effect forecasts
    long_wks = [wks; [wks[end] + Day(7 * k) for k = 1:12]]
    long_mpxv_wkly = [mpxv_wkly; zeros(12, 2)]

    ##Generate main scenarios
    chp_t2 = (Date(2022, 7, 23) - Date(2021, 12, 31)).value #Announcement of Public health emergency
    inf_duration_red = 0.0
    wkly_vaccinations = constants[end-2]

    interventions_ensemble = [
        (
            trans_red2 = θ[9] * θ[11],
            vac_effectiveness = rand(Uniform(0.7, 0.85)),
            trans_red_other2 = θ[10] * θ[12],
            wkly_vaccinations,
            chp_t2,
            inf_duration_red,
        ) for θ in param_draws
    ]

    no_interventions_ensemble = [
        (
            trans_red2 = 0.0,
            vac_effectiveness = 0.0,
            trans_red_other2 = 0.0,
            wkly_vaccinations = zeros(size(long_mpxv_wkly)),
            chp_t2,
            inf_duration_red,
        ) for θ in param_draws
    ]

    no_red_ensemble = [
        (
            trans_red2 = 0.0,
            vac_effectiveness = rand(Uniform(0.7, 0.85)),
            trans_red_other2 = 0.0,
            wkly_vaccinations,
            chp_t2,
            inf_duration_red,
        ) for θ in param_draws
    ]

    no_vac_ensemble = [
        (
            trans_red2 = θ[9] * θ[11],#Based on posterior for first change point with extra dispersion
            vac_effectiveness = rand(Uniform(0.7, 0.85)),
            trans_red_other2 = θ[10] * θ[12],
            wkly_vaccinations = zeros(size(long_mpxv_wkly)),
            chp_t2,
            inf_duration_red,
        ) for θ in param_draws
    ]
    #Simulate
    preds_and_incidence_interventions = map(
        (θ, intervention) ->
            mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention)[2:4],
        param_draws,
        interventions_ensemble,
    )
    preds_and_incidence_no_interventions = map(
        (θ, intervention) ->
            mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention)[2:4],
        param_draws,
        no_interventions_ensemble,
    )
    preds_and_incidence_no_vaccines = map(
        (θ, intervention) ->
            mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention)[2:4],
        param_draws,
        no_vac_ensemble,
    )
    preds_and_incidence_no_redtrans = map(
        (θ, intervention) ->
            mpx_sim_function_interventions(θ, constants, long_mpxv_wkly, intervention)[2:4],
        param_draws,
        no_red_ensemble,
    )

    return (
        preds_and_incidence_interventions = preds_and_incidence_interventions,
        preds_and_incidence_no_interventions = preds_and_incidence_no_interventions,
        preds_and_incidence_no_vaccines = preds_and_incidence_no_vaccines,
        preds_and_incidence_no_redtrans = preds_and_incidence_no_redtrans,
    )
end
"""
    function plot_case_projections(predictions, wks, mpxv_wkly; savefigure=true)

Generate plot of case projections.        
"""
function plot_case_projections(predictions, wks, mpxv_wkly; savefigure = true)
    ## 12 wk lookahead projections
    long_wks = [wks; [wks[end] + Day(7 * k) for k = 1:12]]
    long_mpxv_wkly = [mpxv_wkly; zeros(12, 2)]

    ##Gather data
    preds_and_incidence_interventions,
    preds_and_incidence_no_interventions,
    preds_and_incidence_no_vaccines,
    preds_and_incidence_no_redtrans = predictions
    d1, d2 = size(mpxv_wkly)

    preds = [x[1] for x in preds_and_incidence_interventions]
    preds_nointervention = [x[1] for x in preds_and_incidence_no_interventions]
    preds_novacs = [x[1] for x in preds_and_incidence_no_vaccines]
    preds_noredtrans = [x[1] for x in preds_and_incidence_no_redtrans]
    cum_cases_forwards =
        [cumsum(x[1][(d1+1):end, :], dims = 1) for x in preds_and_incidence_interventions]
    cum_cases_nointervention_forwards = [
        cumsum(x[1][(d1+1):end, :], dims = 1) for x in preds_and_incidence_no_interventions
    ]
    cum_cases_novaccines_forwards =
        [cumsum(x[1][(d1+1):end, :], dims = 1) for x in preds_and_incidence_no_vaccines]
    cum_cases_noredtrans_forwards =
        [cumsum(x[1][(d1+1):end, :], dims = 1) for x in preds_and_incidence_no_redtrans]

    ##Simulation projections

    cred_int = cred_intervals(preds)
    cred_int_rwc = cred_intervals(preds_nointervention)
    cred_int_nv = cred_intervals(preds_novacs)
    cred_int_nr = cred_intervals(preds_noredtrans)

    ## MSM projections
    plt_msm = plot(;
        ylabel = "Weekly cases",
        title = "UK Monkeypox Case Projections (MSM)",# yscale=:log10,
        legend = :topleft,
        # yticks=([1, 2, 11, 101, 1001], [0, 1, 10, 100, 1000]),
        # ylims=(0.8, 3001),
        xticks = (
            [Date(2022, 5, 1) + Month(k) for k = 0:5],
            [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:5],
        ),
        left_margin = 5mm,
        size = (800, 600),
        dpi = 250,
        tickfont = 11,
        titlefont = 18,
        guidefont = 18,
        legendfont = 11,
    )
    plot!(
        plt_msm,
        long_wks,
        cred_int.mean_pred[:, 1],
        ribbon = (cred_int.lb_pred_25[:, 1], cred_int.ub_pred_25[:, 1]),
        lw = 3,
        color = :black,
        fillalpha = 0.2,
        lab = "Forecast",
    )
    plot!(
        plt_msm,
        long_wks[11:end],
        cred_int_nr.mean_pred[11:end, 1],
        ribbon = (cred_int_nr.lb_pred_25[11:end, 1], cred_int_nr.ub_pred_25[11:end, 1]),
        lw = 3,
        ls = :dash,
        color = 1,
        fillalpha = 0.2,
        lab = "No further behavioural response",
    )
    plot!(
        plt_msm,
        long_wks[11:end],
        cred_int_nv.mean_pred[11:end, 1],
        ribbon = (cred_int_nv.lb_pred_25[11:end, 1], cred_int_nv.ub_pred_25[11:end, 1]),
        lw = 3,
        ls = :dash,
        color = 4,
        fillalpha = 0.2,
        lab = "No vaccinations",
    )
    plot!(
        plt_msm,
        long_wks[11:end],
        cred_int_rwc.mean_pred[11:end, 1],
        ribbon = (cred_int_rwc.lb_pred_25[11:end, 1], cred_int_rwc.ub_pred_25[11:end, 1]),
        lw = 3,
        ls = :dash,
        color = 2,
        fillalpha = 0.2,
        lab = "Reasonable worst case scenario",
    )
    scatter!(
        plt_msm,
        wks[(end):end],
        mpxv_wkly[(end):end, 1],
        lab = "",
        ms = 6,
        color = :black,
        shape = :square,
    )
    scatter!(
        plt_msm,
        wks[1:(end-1)],
        mpxv_wkly[1:(end-1), 1],
        lab = "Data",
        ms = 6,
        color = :black,
    )

    ##
    plt_nmsm = plot(;
        ylabel = "Weekly cases",
        title = "UK Monkeypox Case Projections (non-MSM)",# yscale=:log10,
        legend = :topleft,
        # yticks=([1, 2, 11, 101, 1001], [0, 1, 10, 100, 1000]),
        # ylims=(0.8, 3001),
        xticks = (
            [Date(2022, 5, 1) + Month(k) for k = 0:5],
            [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:5],
        ),
        left_margin = 5mm,
        size = (800, 600),
        dpi = 250,
        tickfont = 11,
        titlefont = 18,
        guidefont = 18,
        legendfont = 11,
    )
    plot!(
        plt_nmsm,
        long_wks,
        cred_int.mean_pred[:, 2],
        ribbon = (cred_int.lb_pred_25[:, 2], cred_int.ub_pred_25[:, 2]),
        lw = 3,
        color = :black,
        fillalpha = 0.2,
        lab = "Forecast",
    )
    plot!(
        plt_nmsm,
        long_wks[11:end],
        cred_int_nr.mean_pred[11:end, 2],
        ribbon = (cred_int_nr.lb_pred_25[11:end, 2], cred_int_nr.ub_pred_25[11:end, 2]),
        lw = 3,
        ls = :dash,
        color = 1,
        fillalpha = 0.2,
        lab = "No further behavioural response",
    )
    plot!(
        plt_nmsm,
        long_wks[11:end],
        cred_int_nv.mean_pred[11:end, 2],
        ribbon = (cred_int_nv.lb_pred_25[11:end, 2], cred_int_nv.ub_pred_25[11:end, 2]),
        lw = 3,
        ls = :dash,
        color = 4,
        fillalpha = 0.2,
        lab = "No vaccinations",
    )
    plot!(
        plt_nmsm,
        long_wks[11:end],
        cred_int_rwc.mean_pred[11:end, 2],
        ribbon = (cred_int_rwc.lb_pred_25[11:end, 2], cred_int_rwc.ub_pred_25[11:end, 2]),
        lw = 3,
        ls = :dash,
        color = 2,
        fillalpha = 0.2,
        lab = "Reasonable worst case scenario",
    )
    scatter!(
        plt_nmsm,
        wks[(end):end],
        mpxv_wkly[(end):end, 2],
        lab = "",
        ms = 6,
        color = :black,
        shape = :square,
    )
    scatter!(
        plt_nmsm,
        wks[1:(end-1)],
        mpxv_wkly[1:(end-1), 2],
        lab = "Data",
        ms = 6,
        color = :black,
    )



    ##cumulative Incidence plots
    # cred_int_cum_incidence = cred_intervals(cum_incidences)
    # cred_int_cum_incidence_no_intervention = cred_intervals(cum_incidences_nointervention)
    cred_int_cum_incidence = cred_intervals(cum_cases_forwards)
    cred_int_cum_incidence_no_intervention =
        cred_intervals(cum_cases_nointervention_forwards)
    cred_int_cum_no_vaccines = cred_intervals(cum_cases_novaccines_forwards)
    cred_int_cum_noredtrans = cred_intervals(cum_cases_noredtrans_forwards)


    total_cases = sum(mpxv_wkly, dims = 1)
    plt_cm_msm = plot(;
        ylabel = "Cumulative cases",
        title = "UK Monkeypox cumulative case projections (MSM)",#yscale=:log10,
        legend = :topleft,
        yticks = (0:2500:12500, 0:2500:12500),
        xticks = (
            [Date(2022, 5, 1) + Month(k) for k = 0:5],
            [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:5],
        ),
        left_margin = 5mm,
        size = (800, 600),
        dpi = 250,
        tickfont = 11,
        titlefont = 18,
        guidefont = 18,
        legendfont = 11,
    )
    plot!(
        plt_cm_msm,
        long_wks[((d1+1)):end],
        total_cases[:, 1] .+ cred_int_cum_incidence.mean_pred[:, 1],
        ribbon = (
            cred_int_cum_incidence.lb_pred_25[:, 1],
            cred_int_cum_incidence.ub_pred_25[:, 1],
        ),
        lw = 3,
        color = :black,
        fillalpha = 0.2,
        lab = "Forecast",
    )

    plot!(
        plt_cm_msm,
        long_wks[((d1+1)):end],
        total_cases[:, 1] .+ cred_int_cum_noredtrans.mean_pred[:, 1],
        ribbon = (
            cred_int_cum_noredtrans.lb_pred_25[:, 1],
            cred_int_cum_noredtrans.ub_pred_25[:, 1],
        ),
        lw = 3,
        ls = :dash,
        color = 1,
        fillalpha = 0.2,
        lab = "No further behavioural response",
    )
    plot!(
        plt_cm_msm,
        long_wks[((d1+1)):end],
        total_cases[:, 1] .+ cred_int_cum_no_vaccines.mean_pred[:, 1],
        ribbon = (
            cred_int_cum_no_vaccines.lb_pred_25[:, 1],
            cred_int_cum_no_vaccines.ub_pred_25[:, 1],
        ),
        lw = 3,
        ls = :dash,
        color = 4,
        fillalpha = 0.2,
        lab = "No vaccinations",
    )

    plot!(
        plt_cm_msm,
        long_wks[((d1+1)):end],
        total_cases[:, 1] .+ cred_int_cum_incidence_no_intervention.mean_pred[:, 1],
        ribbon = (
            cred_int_cum_incidence_no_intervention.lb_pred_25[:, 1],
            cred_int_cum_incidence_no_intervention.ub_pred_25[:, 1],
        ),
        lw = 3,
        ls = :dash,
        color = 2,
        fillalpha = 0.2,
        lab = "Reasonable worst case scenario",
    )

    scatter!(
        plt_cm_msm,
        wks,
        cumsum(mpxv_wkly[:, 1], dims = 1),
        lab = "Data",
        ms = 6,
        color = :black,
    )

    ##
    total_cases = sum(mpxv_wkly, dims = 1)
    plt_cm_nmsm = plot(;
        ylabel = "Cumulative cases",
        title = "UK Monkeypox cumulative case projections (non-MSM)",#yscale=:log10,
        legend = :topleft,
        # yticks=(0:2500:12500, 0:2500:12500),
        xticks = (
            [Date(2022, 5, 1) + Month(k) for k = 0:5],
            [monthname(Date(2022, 5, 1) + Month(k))[1:3] for k = 0:5],
        ),
        left_margin = 5mm,
        size = (800, 600),
        dpi = 250,
        tickfont = 11,
        titlefont = 17,
        guidefont = 18,
        legendfont = 11,
    )
    plot!(
        plt_cm_nmsm,
        long_wks[((d1+1)):end],
        total_cases[:, 2] .+ cred_int_cum_incidence.mean_pred[:, 2],
        ribbon = (
            cred_int_cum_incidence.lb_pred_25[:, 2],
            cred_int_cum_incidence.ub_pred_25[:, 2],
        ),
        lw = 3,
        color = :black,
        fillalpha = 0.2,
        lab = "Forecast",
    )

    plot!(
        plt_cm_nmsm,
        long_wks[((d1+1)):end],
        total_cases[:, 2] .+ cred_int_cum_noredtrans.mean_pred[:, 2],
        ribbon = (
            cred_int_cum_noredtrans.lb_pred_25[:, 2],
            cred_int_cum_noredtrans.ub_pred_25[:, 2],
        ),
        lw = 3,
        ls = :dash,
        color = 1,
        fillalpha = 0.2,
        lab = "No further behavioural response",
    )
    plot!(
        plt_cm_nmsm,
        long_wks[((d1+1)):end],
        total_cases[:, 2] .+ cred_int_cum_no_vaccines.mean_pred[:, 2],
        ribbon = (
            cred_int_cum_no_vaccines.lb_pred_25[:, 2],
            cred_int_cum_no_vaccines.ub_pred_25[:, 2],
        ),
        lw = 3,
        ls = :dash,
        color = 4,
        fillalpha = 0.2,
        lab = "No vaccinations",
    )

    plot!(
        plt_cm_nmsm,
        long_wks[((d1+1)):end],
        total_cases[:, 2] .+ cred_int_cum_incidence_no_intervention.mean_pred[:, 2],
        ribbon = (
            cred_int_cum_incidence_no_intervention.lb_pred_25[:, 2],
            cred_int_cum_incidence_no_intervention.ub_pred_25[:, 2],
        ),
        lw = 3,
        ls = :dash,
        color = 2,
        fillalpha = 0.2,
        lab = "Reasonable worst case scenario",
    )

    scatter!(
        plt_cm_nmsm,
        wks,
        cumsum(mpxv_wkly[:, 2], dims = 1),
        lab = "Data",
        ms = 6,
        color = :black,
    )

    ##Combined plot for cases
    lo = @layout [a b; c d]
    plt = plot(
        plt_msm,
        plt_nmsm,
        plt_cm_msm,
        plt_cm_nmsm,
        size = (1600, 1200),
        dpi = 250,
        left_margin = 10mm,
        bottom_margin = 10mm,
        right_margin = 10mm,
        layout = lo,
    )
    if savefigure
        savefig(plt, "plots/case_projections_" * string(wks[end]) * ".png")
    end
    return plt
end
