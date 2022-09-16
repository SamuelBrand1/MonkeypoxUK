using CSV, DataFrames, Plots, Dates, Plots.PlotMeasures

## MSM data with data inference

mpxv_data_inferred = CSV.File("data/mxpv_wkly_inferredmsm.csv") |> DataFrame
wks = Date.(mpxv_data_inferred.week, DateFormat("dd/mm/yyyy"))
mpxv_wkly = [mpxv_data_inferred.gbmsm mpxv_data_inferred.nongbmsm]
