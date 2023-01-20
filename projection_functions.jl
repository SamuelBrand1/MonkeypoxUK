"""
    function load_posteriors_for_projection(date_str, description_str; pheic_effect = true)

Load the necessary saved posterior data from simulations to make forecasts.        
"""
function load_posteriors_for_projection(date_str, description_str; pheic_effect = true)
    param_draws =
        load("posteriors/posterior_param_draws_" * date_str * description_str * ".jld2")["param_draws"]
    detected_cases =
        load("posteriors/posterior_detected_cases_" * date_str * description_str * ".jld2")["detected_cases"]
    vac_effs = load(
        "posteriors/posterior_vac_effectivenesses_" * date_str * description_str * ".jld2",
    )["vac_effectivenesses"]
    end_states =
        load("posteriors/posterior_end_states_" * date_str * description_str * ".jld2")["end_states"]
    
    if !pheic_effect
        param_draws = map(θ -> vcat(θ[1:10], 0.0, 0.0), deepcopy(param_draws))
    end

    return (; param_draws, detected_cases, vac_effs, end_states)
end

"""
    function add_proj_plot(plt_gbmsm, plt_nongbmsm, post_draws, start_wk, clr, lab_str ; n_samples = 2000)

Performs a projection from posterior data from `post_draws`, and adds projections to plots. Also calculates the median relative error
over each week with available data.        
"""
function add_proj_plot(plt_gbmsm, plt_nongbmsm, post_draws, start_wk, clr, lab_str, n_vac ; n_samples = 2000)
    proj_fromend = [
        (
            ts = [start_wk + Week(k) for k = 1:12] .|> d -> (d - Date(2021, 12, 31)).value |> Float64,
            wkly_vaccinations = wkly_vaccinations[(n_vac+1):end],
            vac_effectiveness = post_draws.vac_effs[k],
        ) for k = 1:n_samples
    ]

    projections_from_end = @showprogress 0.1 "MPX Forecasts:" map(
        (θ, interventions, state) ->
            mpx_sim_function_projections(θ, constants, interventions, state),
        post_draws.param_draws,
        proj_fromend,
        post_draws.end_states,
    )

    cred_proj = MonkeypoxUK.cred_intervals(
        [proj.detected_cases for proj in projections_from_end],
        central_est = :median,
    )

    cred_prev_cases = MonkeypoxUK.cred_intervals(
        [proj for proj in post_draws.detected_cases],
        central_est = :median,
    )
    
    n_wks = size(cred_prev_cases.median_pred, 1)
    past_wks = (start_wk - Week(n_wks-1)):Week(1):(start_wk) |> collect
    

    plot!(plt_gbmsm, 
            title = "GBMSM case proj. from " * string(start_wk),
            )

    plot!(plt_nongbmsm,
        title = "non-GBMSM case proj. from " * string(start_wk),
        )        

    plot!(plt_gbmsm, past_wks, cred_prev_cases.median_pred[:,1],
        lab = "",
        c= clr,
        lw = 2,
        ls = :dot,
        )

    plot!(plt_gbmsm, [start_wk + Week(k) for k = 1:12], cred_proj.median_pred[:,1],
           ribbon = (cred_proj.lb_pred_25[:,1], cred_proj.ub_pred_25[:,1]),
           lab = lab_str,
           c= clr,
           lw = 3,
           fillalpha = 0.2) 

    plot!(plt_nongbmsm, past_wks, cred_prev_cases.median_pred[:,2],
           lab = "",
           c= clr,
           lw = 2,
           ls = :dot,
           )
   

    plot!(plt_nongbmsm, [start_wk + Week(k) for k = 1:12], cred_proj.median_pred[:,2],
            ribbon = (cred_proj.lb_pred_25[:,2], cred_proj.ub_pred_25[:,2]),
            lab = lab_str,
            c= clr,
            lw = 3,
            fillalpha = 0.2) 
            
    f1 = findall([wk ∈ wks for wk in [start_wk + Week(k) for k = 1:12]])
    f2 = findall([wk ∈ [start_wk + Week(k) for k = 1:12] for wk in wks])


    if !isempty(f1)
        errors = [sum(abs, proj_cases.detected_cases[f1,:] .- mpxv_wkly[f2,:]) ./ sum(mpxv_wkly[f2,:])  for proj_cases in projections_from_end]
        err = median(errors)
        err_range = quantile(errors, [0.025, 0.975])
        median_forecast_err = sum(abs, cred_proj.median_pred[f1, :] .- mpxv_wkly[f2,:])  ./ sum(mpxv_wkly[f2,:])
        return (err, err_range), median_forecast_err   
    else
        return (0.0, (0.0,0.0)), 0.0
    end
    
end

function load_data_and_make_proj(start_wk, description_str, plt_gbmsm, plt_nongbmsm, clr, lab_str, n_vac ; n_samples = 2000, pheic_effect = true)
    post_draws = load_posteriors_for_projection(string(start_wk), description_str; pheic_effect = pheic_effect)    
    add_proj_plot(plt_gbmsm, plt_nongbmsm, post_draws, start_wk, clr, lab_str, n_vac; n_samples = n_samples)
end
