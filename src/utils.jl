function sigmoid(x)
    1.0 / (1.0 + exp(-x))
end

"""
    function cred_intervals(preds)

Generate posterior mean, median and 25% and 2.5% (relative to posterior mean) predictions from a posterior sample array over two-column matrices that 
represent sampled weekly incidence of MSM and non-MSM people. Output a `NamedTuple` object.
"""
function cred_intervals(preds; central_est = :mean)
    @assert (central_est == :mean || central_est == :median) "Central estimate choice must be mean or median."

    mean_pred = hcat(
        [mean([preds[n][wk, 1] for n = 1:length(preds)]) for wk = 1:size(preds[1], 1)],
        [mean([preds[n][wk, 2] for n = 1:length(preds)]) for wk = 1:size(preds[1], 1)],
    )
    median_pred = hcat(
        [median([preds[n][wk, 1] for n = 1:length(preds)]) for wk = 1:size(preds[1], 1)],
        [median([preds[n][wk, 2] for n = 1:length(preds)]) for wk = 1:size(preds[1], 1)],
    )

    if central_est == :mean
        lb_pred_25 =
            mean_pred .- hcat(
                [
                    quantile([preds[n][wk, 1] for n = 1:length(preds)], 0.25) for
                    wk = 1:size(preds[1], 1)
                ],
                [
                    quantile([preds[n][wk, 2] for n = 1:length(preds)], 0.25) for
                    wk = 1:size(preds[1], 1)
                ],
            )

        lb_pred_10 =
            mean_pred .- hcat(
                [
                    quantile([preds[n][wk, 1] for n = 1:length(preds)], 0.10) for
                    wk = 1:size(preds[1], 1)
                ],
                [
                    quantile([preds[n][wk, 2] for n = 1:length(preds)], 0.10) for
                    wk = 1:size(preds[1], 1)
                ],
            )

        lb_pred_025 =
            mean_pred .- hcat(
                [
                    quantile([preds[n][wk, 1] for n = 1:length(preds)], 0.025) for
                    wk = 1:size(preds[1], 1)
                ],
                [
                    quantile([preds[n][wk, 2] for n = 1:length(preds)], 0.025) for
                    wk = 1:size(preds[1], 1)
                ],
            )

        ub_pred_25 =
            hcat(
                [
                    quantile([preds[n][wk, 1] for n = 1:length(preds)], 0.75) for
                    wk = 1:size(preds[1], 1)
                ],
                [
                    quantile([preds[n][wk, 2] for n = 1:length(preds)], 0.75) for
                    wk = 1:size(preds[1], 1)
                ],
            ) .- mean_pred

        ub_pred_10 =
            hcat(
                [
                    quantile([preds[n][wk, 1] for n = 1:length(preds)], 0.9) for
                    wk = 1:size(preds[1], 1)
                ],
                [
                    quantile([preds[n][wk, 2] for n = 1:length(preds)], 0.9) for
                    wk = 1:size(preds[1], 1)
                ],
            ) .- mean_pred

        ub_pred_025 =
            hcat(
                [
                    quantile([preds[n][wk, 1] for n = 1:length(preds)], 0.975) for
                    wk = 1:size(preds[1], 1)
                ],
                [
                    quantile([preds[n][wk, 2] for n = 1:length(preds)], 0.975) for
                    wk = 1:size(preds[1], 1)
                ],
            ) .- mean_pred
        return (
            mean_pred = mean_pred,
            median_pred = median_pred,
            lb_pred_025 = lb_pred_025,
            lb_pred_10 = lb_pred_10,
            lb_pred_25 = lb_pred_25,
            ub_pred_25 = ub_pred_25,
            ub_pred_10 = ub_pred_10,
            ub_pred_025 = ub_pred_025,
        )
    else
        lb_pred_25 =
            median_pred .- hcat(
                [
                    quantile([preds[n][wk, 1] for n = 1:length(preds)], 0.25) for
                    wk = 1:size(preds[1], 1)
                ],
                [
                    quantile([preds[n][wk, 2] for n = 1:length(preds)], 0.25) for
                    wk = 1:size(preds[1], 1)
                ],
            )

        lb_pred_10 =
            median_pred .- hcat(
                [
                    quantile([preds[n][wk, 1] for n = 1:length(preds)], 0.10) for
                    wk = 1:size(preds[1], 1)
                ],
                [
                    quantile([preds[n][wk, 2] for n = 1:length(preds)], 0.10) for
                    wk = 1:size(preds[1], 1)
                ],
            )

        lb_pred_025 =
            median_pred .- hcat(
                [
                    quantile([preds[n][wk, 1] for n = 1:length(preds)], 0.025) for
                    wk = 1:size(preds[1], 1)
                ],
                [
                    quantile([preds[n][wk, 2] for n = 1:length(preds)], 0.025) for
                    wk = 1:size(preds[1], 1)
                ],
            )

        ub_pred_25 =
            hcat(
                [
                    quantile([preds[n][wk, 1] for n = 1:length(preds)], 0.75) for
                    wk = 1:size(preds[1], 1)
                ],
                [
                    quantile([preds[n][wk, 2] for n = 1:length(preds)], 0.75) for
                    wk = 1:size(preds[1], 1)
                ],
            ) .- median_pred

        ub_pred_10 =
            hcat(
                [
                    quantile([preds[n][wk, 1] for n = 1:length(preds)], 0.9) for
                    wk = 1:size(preds[1], 1)
                ],
                [
                    quantile([preds[n][wk, 2] for n = 1:length(preds)], 0.9) for
                    wk = 1:size(preds[1], 1)
                ],
            ) .- median_pred

        ub_pred_025 =
            hcat(
                [
                    quantile([preds[n][wk, 1] for n = 1:length(preds)], 0.975) for
                    wk = 1:size(preds[1], 1)
                ],
                [
                    quantile([preds[n][wk, 2] for n = 1:length(preds)], 0.975) for
                    wk = 1:size(preds[1], 1)
                ],
            ) .- median_pred

        return (
            mean_pred = mean_pred,
            median_pred = median_pred,
            lb_pred_025 = lb_pred_025,
            lb_pred_10 = lb_pred_10,
            lb_pred_25 = lb_pred_25,
            ub_pred_25 = ub_pred_25,
            ub_pred_10 = ub_pred_10,
            ub_pred_025 = ub_pred_025,
        )

    end
end

"""
    function matrix_cred_intervals(preds)

Generate posterior mean, median and 25% and 2.5% (relative to posterior mean) predictions from a posterior sample array over n-column matrices. Output a `NamedTuple` object.
"""
function matrix_cred_intervals(preds; central_est = :mean)
    @assert (central_est == :mean || central_est == :median) "Central estimate choice must be `:mean` or `:median`."

    d1, d2 = size(preds[1])
    num = length(preds)
    median_pred = Matrix{Float64}(undef, d1, d2)
    mean_pred = similar(median_pred)
    lb_pred_25 = similar(median_pred)
    lb_pred_025 = similar(median_pred)
    lb_pred_10 = similar(median_pred)
    ub_pred_25 = similar(median_pred)
    ub_pred_025 = similar(median_pred)
    ub_pred_10 = similar(median_pred)

    if central_est == :mean
        for i = 1:d1, j = 1:d2
            v = [preds[n][i, j] for n = 1:num]
            median_pred[i, j] = median(v)
            mean_pred[i, j] = mean(v)
            lb_pred_25[i, j] = quantile(v, 0.25)
            lb_pred_025[i, j] = quantile(v, 0.025)
            lb_pred_10[i, j] = quantile(v, 0.1)
            ub_pred_25[i, j] = quantile(v, 0.75)
            ub_pred_025[i, j] = quantile(v, 0.975)
            ub_pred_10[i, j] = quantile(v, 0.9)
        end
        lb_pred_25 .= mean_pred .- lb_pred_25
        lb_pred_025 .= mean_pred .- lb_pred_025
        lb_pred_10 .= mean_pred .- lb_pred_10

        ub_pred_25 .= ub_pred_25 .- mean_pred
        ub_pred_025 .= ub_pred_025 .- mean_pred
        ub_pred_10 .= ub_pred_10 .- mean_pred

        return (
            median_pred = median_pred,
            mean_pred = mean_pred,
            lb_pred_10 = lb_pred_10,
            lb_pred_025 = lb_pred_025,
            lb_pred_25 = lb_pred_25,
            ub_pred_25 = ub_pred_25,
            ub_pred_025 = ub_pred_025,
            ub_pred_10 = ub_pred_10,
        )
    else
        for i = 1:d1, j = 1:d2
            v = [preds[n][i, j] for n = 1:num]
            median_pred[i, j] = median(v)
            mean_pred[i, j] = mean(v)
            lb_pred_25[i, j] = quantile(v, 0.25)
            lb_pred_025[i, j] = quantile(v, 0.025)
            lb_pred_10[i, j] = quantile(v, 0.1)
            ub_pred_25[i, j] = quantile(v, 0.75)
            ub_pred_025[i, j] = quantile(v, 0.975)
            ub_pred_10[i, j] = quantile(v, 0.9)
        end
        lb_pred_25 .= median_pred .- lb_pred_25
        lb_pred_025 .= median_pred .- lb_pred_025
        lb_pred_10 .= median_pred .- lb_pred_10

        ub_pred_25 .= ub_pred_25 .- median_pred
        ub_pred_025 .= ub_pred_025 .- median_pred
        ub_pred_10 .= ub_pred_10 .- median_pred

        return (
            median_pred = median_pred,
            mean_pred = mean_pred,
            lb_pred_10 = lb_pred_10,
            lb_pred_025 = lb_pred_025,
            lb_pred_25 = lb_pred_25,
            ub_pred_25 = ub_pred_25,
            ub_pred_025 = ub_pred_025,
            ub_pred_10 = ub_pred_10,
        )
    end

end

"""
    function mom_fit_beta(X; shrinkage = 1.0, bias_factor = 1.0)

    Fit future change in risk based on posterior for first change point with extra dispersion due to `shrinkage`,
        and possible bias due to `bias_factor`.
"""
function mom_fit_beta(X; shrinkage = 1.0, bias_factor = 1.0)
    x̄ = mean(X)
    v̄ = var(X)
    if v̄ < x̄ * (1 - x̄)
        α = ((x̄^2 * (1 - x̄) / v̄) - x̄) / shrinkage
        β = (((x̄ * (1 - x̄)^2) / v̄) - (1 - x̄)) / shrinkage
        α̂ = bias_factor * α
        β̂ = (1 - bias_factor) * α + β
        return Beta(α̂, β̂)
    else
        println("ERROR")
        return nothing
    end
end
