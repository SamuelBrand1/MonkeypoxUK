# MonkeypoxUK

The `MonkeypoxUK` module provides methods for simulating MPX spread among men-who-have-sex-with-men (MSM) in the United Kingdom as well as the wider community. Weekly case data is from a combination of [Global.Health/ourworldindata](https://ourworldindata.org/monkeypox) and [UKHSA technical briefings](https://www.gov.uk/government/publications/monkeypox-outbreak-technical-briefings).

A first preprint describing the underlying reasoning and methodology is now available [_The role of vaccination and public awareness in medium-term forecasts of monkeypox incidence in the United Kingdom_](https://www.medrxiv.org/content/10.1101/2022.08.15.22278788v1).

A second preprint using data directly from the UKHSA, rather than open source data from Global.Health, and with an updated set of counter-factual scenarios is now available [_The role of vaccination and public awareness in forecasts of monkeypox incidence in the United Kingdom_](https://www.researchsquare.com/article/rs-2162921/v1).

### Quick start for inference

1. Download [Julia](https://julialang.org/downloads/).
2. Clone this repository.
3. Start the Julia REPL.
4. Change working directory to where this repo is cloned.
5. Enter `Pkg` mode by pressing `]`
6. Activate the environment for `MonkeypoxUK` and download the underlying dependencies.
    > pkg> activate . \
    > pkg> instantiate
7. The script `mpx_inference.jl` covers running the inference methodology. The script `mpxv_datawrangling.jl` loads the underlying case data into a two column matrix `mpxv_wkly` where rows are weeks and first column is reported MSM cases and second column is reported non-MSM cases. The Monday date for each week is given as a `Vector{Date}` array `wks`.

### Latest case projections for the UK

<figure>
<img src="plots/case_projections_2022-10-17.png" alt="Trulli" style="width:100%">
<figcaption align = "center"><b>Posterior means and 10-90% posterior probabilities </b></figcaption>
</figure>


### Method Update [15-09-2022]

We are transitioning to using a method for inferring GBMSM proportion among non-reporting individuals in the UKHSA reporting dataset. For results using the datasets described above (and in linked preprint) please refer to `\plots_globalhealth` and `\posteriors_globalhealth`. 

### Data update [26-09-2022]

The data set from Global.Health has now depreciated.