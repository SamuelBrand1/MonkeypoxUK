"""
    function simulation_based_calibration(prior_vect, wks, mpxv_wkly, constants; savefigure=true, verbose=true, target_perc=0.05)

Run a prior predictive check for the MPX model and output as a plot. Also run model-based calibration of errors at a target
    percentile error `target_perc` (default = 5%), and return as `ϵ_target`.        
"""
function simulation_based_calibration(
    prior_vect,
    wks,
    mpxv_wkly,
    constants;
    savefigure = true,
    verbose = true,
    target_perc = 0.05,
)
    if verbose
        println("Starting prior predictive checking simulations")
    end
    ## Prior predictive checking - simulation
    draws = [rand.(prior_vect) for i = 1:1000]
    prior_sims = map(θ -> MonkeypoxUK.mpx_sim_function_chp(θ, constants, mpxv_wkly), draws)

    ##Prior predictive checking - simulation
    prior_preds = [sim[2].detected_cases for sim in prior_sims]
    plt_priorpredcheck =
        plot(; ylabel = "Weekly cases", title = "Prior predictive checking")
    for pred in prior_preds
        plot!(plt_priorpredcheck, wks, pred, lab = "", color = [1 2], alpha = 0.3)
    end
    if savefigure
        savefig(
            plt_priorpredcheck,
            "plots/prior_predictive_checking_plot" * string(wks[end]) * ".png",
        )
    end
    ## Model-based calibration of target tolerance
    if verbose
        println("Starting model-based calibration")
    end
    mbc_errs = map(
        n -> MonkeypoxUK.mpx_sim_function_chp(
            draws[n],
            constants,
            prior_sims[n][2].detected_cases,
        )[1],
        1:1000,
    )

    ##Find target tolerance and plot error distribution
    ϵ_target = find_zero(x -> target_perc - sum(mbc_errs .< x) / length(mbc_errs), (0, 5))
    err_hist = histogram(
        mbc_errs,
        norm = :pdf,
        nbins = 500,
        lab = "",
        title = "Sampled errors from simulations with exact parameters",
        xlabel = "L1 relative error",
        xlims = (0, 5),
        size = (700, 400),
    )
    vline!(
        err_hist,
        [ϵ_target],
        lab = "$(round(Int64,target_perc*100))th percentile (target err. = $(round(ϵ_target,digits = 3)))",
        lw = 3,
    )
    if savefigure
        savefig(err_hist, "plots/mbc_error_calibration_plt" * string(wks[end]) * ".png")
    end
    return ϵ_target, plt_priorpredcheck, err_hist
end

function convert_to_chn(smc)
    ##Create transformations to more interpetable parameters
    param_names = [
        :clique_dispersion,
        :prob_detect,
        :mean_inf_period,
        :prob_transmission,
        :R0_other,
        :detect_dispersion,
        :init_infs,
        :chg_pnt,
        :sex_trans_red,
        :other_trans_red,
        :sex_trans_red_post_WHO,
        :other_trans_red_post_WHO,
    ]

    transformations = [
        fill(x -> x, 2)
        x -> 1 + mean(Geometric(1 / (1 + x))) # Translate the infectious period parameter into mean infectious period
        fill(x -> x, 2)
        x -> 1 / (x + 1) #Translate "effective sample size" for Beta-Binomial on sampling to overdispersion parameter
        fill(x -> x, 4)
        fill(x -> x, 2)
    ]
    function col_transformations(X, f_vect)
        for j = 1:size(X, 2)
            X[:, j] = f_vect[j].(X[:, j])
        end
        return X
    end

    val_mat =
        smc.parameters |>
        X ->
            col_transformations(X, transformations) |>
            X ->
                hcat(X[:, 1:10], X[:, 11] .* X[:, 4], X[:, 12] .* X[:, 5]) |>
                X -> [X[i, j] for i = 1:size(X, 1), j = 1:size(X, 2), k = 1:1]
    chn = Chains(val_mat, param_names)

    untransformed_chain
    # write("posteriors/posterior_chain_" * string(wks[end]) * ".jls", chn)
    return chn
end

function load_smc(filename)
    return load(filename)["smc_cng_pnt"]
end
