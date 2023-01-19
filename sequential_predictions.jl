using Distributions, StatsBase, StatsPlots, Plots.PlotMeasures
using LinearAlgebra, RecursiveArrayTools, CSV
using OrdinaryDiffEq, ApproxBayes, DataFrames
using JLD2, MCMCChains, ProgressMeter
using MonkeypoxUK
using ColorSchemes, Dates
using Latexify

## MSM data with data inference

past_mpxv_data_inferred =
    CSV.File("data/weekly_data_imputation_2022-09-30.csv", missingstring = "NA") |>
    DataFrame

wks = Date.(past_mpxv_data_inferred.week, DateFormat("dd/mm/yyyy"))

include("setup_model.jl");

## Format case data into expected (over GBDT inference) case incidence

colname = "seqn_fit5"

inferred_prop_na_msm = past_mpxv_data_inferred[:, colname] |> x -> x[.~ismissing.(x)]
inferred_prop_na_msm_lwr =
    past_mpxv_data_inferred[:, "lower_"*colname] |> x -> x[.~ismissing.(x)]
inferred_prop_na_msm_upr =
    past_mpxv_data_inferred[:, "upper_"*colname] |> x -> x[.~ismissing.(x)]

mpxv_wkly =
    past_mpxv_data_inferred[1:size(inferred_prop_na_msm, 1), ["gbmsm", "nongbmsm"]] .+
    past_mpxv_data_inferred[1:size(inferred_prop_na_msm, 1), "na_gbmsm"] .*
    hcat(inferred_prop_na_msm, 1.0 .- inferred_prop_na_msm) |> Matrix

lwr_mpxv_wkly =
    past_mpxv_data_inferred[1:size(inferred_prop_na_msm, 1), ["gbmsm", "nongbmsm"]] .+
    past_mpxv_data_inferred[1:size(inferred_prop_na_msm, 1), "na_gbmsm"] .*
    hcat(inferred_prop_na_msm_lwr, 1.0 .- inferred_prop_na_msm_lwr) |> Matrix

upr_mpxv_wkly =
    past_mpxv_data_inferred[1:size(inferred_prop_na_msm, 1), ["gbmsm", "nongbmsm"]] .+
    past_mpxv_data_inferred[1:size(inferred_prop_na_msm, 1), "na_gbmsm"] .*
    hcat(inferred_prop_na_msm_upr, 1.0 .- inferred_prop_na_msm_upr) |> Matrix



##

wks = Date.(past_mpxv_data_inferred.week[1:size(mpxv_wkly, 1)], DateFormat("dd/mm/yyyy"))
ts = wks .|> d -> d - Date(2021, 12, 31) .|> t -> t.value

wkly_vaccinations = [
    [zeros(12); 1000; 2000; fill(5000, 4)] * 1.675
    fill(650, 18)
]

## Generate an ensemble of forecasts

seq_wks = [wks[1:4], wks[1:8], wks[1:12], wks[1:16], wks[1:20]]

seq_mpxv_wklys = [
    mpxv_wkly[1:4, :],
    mpxv_wkly[1:8, :],
    mpxv_wkly[1:12, :],
    mpxv_wkly[1:16, :],
    mpxv_wkly[1:20, :],
    # mpxv_wkly,
]

## Include useful functions for projections 

include("projection_functions.jl");

##

description_strs = ["no_ngbmsm_chg", "", "no_bv_cng", "one_metapop"]
description_labs = ["Main model", "Also non-GBMSM behaviour change", "No behaviour change", "Homo. pop."]
clrs = 1:4

errs_by_data = map(1:5) do n
    n_vac = (length(seq_wks[n])+1)
    proj_weeks = seq_wks[n]
    start_wk = proj_weeks[end] 
    plt_gbmsm = plot(;
                    ylabel = "Weekly cases",
                    legend = :topright,
                    left_margin = 5mm,
                    right_margin = 5mm,
                    size = (800, 600),
                    dpi = 250,
                    ytickfont = 18,
                    xtickfont = 12,
                    titlefont = 20,
                    guidefont = 24,
                    legendfont = 12)


    plt_nongbmsm = deepcopy(plt_gbmsm)

    if n >= 4
        plot!(plt_gbmsm; ylims = (0,650))
        plot!(plt_nongbmsm; ylims = (0,65))
    end

    err_by_model = map((description_str, clr, description_lab) -> load_data_and_make_proj(start_wk, description_str, plt_gbmsm, plt_nongbmsm, clr, description_lab, n_vac; pheic_effect = n > 2),
                        description_strs,
                        clrs,
                        description_labs)

    scatter!(plt_gbmsm, wks, mpxv_wkly[:,1],
            lab = "Data available (6th Oct 2022)",
            ms = 6,
            color = :black,
            yerrors = (
                mpxv_wkly[:, 1] .- lwr_mpxv_wkly[:, 1],
                upr_mpxv_wkly[:, 1] .- mpxv_wkly[:, 1],
            ),)                    

    scatter!(plt_nongbmsm, wks, mpxv_wkly[:,2],
                lab = "Data available (6th Oct 2022)",
                ms = 6,
                color = :black,
                yerrors = (
                    mpxv_wkly[:, 2] .- lwr_mpxv_wkly[:, 2],
                    upr_mpxv_wkly[:, 2] .- mpxv_wkly[:, 2],
                ),)                    

    plt = plot(plt_gbmsm, plt_nongbmsm,
                size = (1500,600),
                dpi = 250,
                left_margin = 10mm,
                right_margin = 0mm,
                bottom_margin = 5mm)

    savefig(plt, "plots/proj_plot_" * string(start_wk) * ".png")
    
    return err_by_model
end

## Past fits



##

df_errors = DataFrame(date = String[], 
                        main_model_median_error = String[],
                        full_model_median_error = String[],
                        no_behaviour_change_median_error = String[],
                        one_metapopulation_median_error = String[],
                        main_model_forecast_err = Number[],
                        full_model_forecast_err = Number[],
                        no_behaviour_change_forecast_err = Number[],
                        one_metapopulation_forecast_err = Number[])

for k = 1:4                        
    push!(df_errors, 
            [string(seq_wks[k][end]);
            [string(errs_by_data[k][n][1])[2:(end-1)] for n = 1:4];
            [errs_by_data[k][n][2] for n = 1:4]],
            )                        
end

CSV.write("projections/forecast_errors.csv", df_errors)
# project_errors_tex = latexify(df_errors, env = :table)

# output_tex = raw"\newcommand{\projectiontable}{" * project_errors_tex * raw"}"

# open("model_output.tex"; append = true) do io
#     write(io, output_tex)
# end;