#set wd in windows:
#cd(raw"C:\Users\massimo.cavallaro\Documents\MPX\MonkeypoxUK\\")

using CSV, DataFrames, Plots, Dates, Plots.PlotMeasures, ColorSchemes

#data = CSV.File("X:/analysis/csv/weekly_data_imputation_2022-09-30.csv") |> DataFrame;
data = CSV.File("X:/analysis/csv/0.75_sequential_2022-10-09.csv") |> DataFrame;
color = get(ColorSchemes.cool, [18, 17, 13, 9, 5] / (size(data, 1) - 2))
idx = data.X26 .!= "NA";
fillalpha = 0.3
lw = 3
#plot(data.week[idx], parse.(Float64, data.X26[idx]),
plot(
    data.week[idx],
    data.X26[idx],
    ylabel = "p(w)",
    xlabel = "w",
    label = "26/09/2022",
    ylim = [0.5, 1],
    color = color[1],
    lw = lw,
    legend = :bottomleft,
)
#plot!(data.week[idx], parse.(Float64, data[:,"lower.26"][idx]),
plot!(
    data.week[idx],
    data[:, "lower.26"][idx],
    #fillrange = parse.(Float64, data[:,"upper.26"][idx]), label="",
    fillrange = data[:, "upper.26"][idx],
    label = "",
    fillalpha = fillalpha,
    color = color[1],
    lw = 0,
)

idx = data.X24 .!= "NA";
plot!(
    data.week[idx],
    parse.(Float64, data.X24[idx]),
    color = color[2],
    label = "12/09/2022",
    lw = lw,
)
plot!(
    data.week[idx],
    parse.(Float64, data[:, "lower.24"][idx]),
    fillrange = parse.(Float64, data[:, "upper.24"][idx]),
    label = "",
    color = color[2],
    fillalpha = fillalpha,
    lw = 0,
)

idx = data.X20 .!= "NA";
plot!(
    data.week[idx],
    parse.(Float64, data.X20[idx]),
    color = color[3],
    label = "15/08/2022",
    lw = lw,
)
plot!(
    data.week[idx],
    parse.(Float64, data[:, "lower.20"][idx]),
    fillrange = parse.(Float64, data[:, "upper.20"][idx]),
    label = "",
    color = color[3],
    fillalpha = fillalpha,
    lw = 0,
)

idx = data.X16 .!= "NA";
plot!(
    data.week[idx],
    parse.(Float64, data.X16[idx]),
    color = color[4],
    label = "18/07/2022",
    lw = lw,
)
plot!(
    data.week[idx],
    parse.(Float64, data[:, "lower.16"][idx]),
    fillrange = parse.(Float64, data[:, "upper.16"][idx]),
    label = "",
    color = color[4],
    fillalpha = fillalpha,
    lw = 0,
)

idx = data.X12 .!= "NA";
plot!(
    data.week[idx],
    parse.(Float64, data.X12[idx]),
    color = color[5],
    label = "20/06/2022",
    lw = lw,
)
plot!(
    data.week[idx],
    parse.(Float64, data[:, "lower.12"][idx]),
    fillrange = parse.(Float64, data[:, "upper.12"][idx]),
    label = "",
    color = color[5],
    fillalpha = fillalpha,
    lw = 0,
)

savefig(
    "C:\\Users\\massimo.cavallaro\\Documents\\MPX\\0.75_sequential_imputation_9_10_22.png",
)
