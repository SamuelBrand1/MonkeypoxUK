using CSV, DataFrames, Plots, Dates, Plots.PlotMeasures

## MSM data with data inference

past_mpxv_data_inferred = CSV.File("data/weekly_data_imputation_2022-10-26.csv",
                                missingstring = "NA") |> DataFrame

wks = Date.(past_mpxv_data_inferred.week, DateFormat("dd/mm/yyyy"))