# MonkeypoxUK

The `MonkeypoxUK` module provides methods for simulating MPX spread among Gay-and-Bisexual-men-who-have-sex-with-men (GBMSM) in the United Kingdom as well as the wider community. Weekly case data is from a combination of [Global.Health/ourworldindata](https://ourworldindata.org/monkeypox) and [UKHSA technical briefings](https://www.gov.uk/government/publications/monkeypox-outbreak-technical-briefings).

A first preprint describing the underlying reasoning and methodology is available [_The role of vaccination and public awareness in medium-term forecasts of monkeypox incidence in the United Kingdom_](https://www.medrxiv.org/content/10.1101/2022.08.15.22278788v1). The code base for this has now been removed in favour of new techniques. It can be recovered via looking at the history of commits on this repository.

A second preprint using data directly from the UKHSA, rather than open source data from Global.Health, and with an updated set of counter-factual scenarios is also available [_The role of vaccination and public awareness in forecasts of monkeypox incidence in the United Kingdom_](https://www.researchsquare.com/article/rs-2162921/v1).

### Quick start for inference

1. Download [Julia](https://julialang.org/downloads/).
2. Clone this repository.
3. Start the Julia REPL.
4. Change working directory to where this repo is cloned.
5. Enter `Pkg` mode by pressing `]`
6. Activate the environment for `MonkeypoxUK` and download the underlying dependencies.
    > pkg> activate . \
    > pkg> instantiate
7. The script `mpx_inference.jl` covers running the inference methodology. This also loads the underlying case data into a two column matrix `mpxv_wkly` where rows are weeks and first column is reported MSM cases and second column is reported non-MSM cases. The Monday date for each week is given as a `Vector{Date}` array `wks`.

### Other scripts in this repository

* `mpx_model_checking.jl`. Creates a PIT histogram and QQ plot for the main model case projections.
* `mpx_projections.jl`. Generates the main figures.
* `posterior_plots.jl`. Generates figures for posterior distribution of model parameters.
* `sequential_predictions.j`. Generates alternate model comparisons.
