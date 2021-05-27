# CS 156b - Analysis of Policy Effectiveness and Time-Sensitivity of Policy Implementation on COVID-19 Output Metrics

## disco_panda Team Members

    Thai Khong - Did some initial data analysis regarding effects of policy changes on COVID-19 output metrics. Focused on analyzing policy effectiveness using the Facebook Prophet models (with linear and logistic growth curve)

    Sasha Fomina -

    Beau Lobodin -

# Models
## Facebook Prophet - Time Series Model
### References:
1. [Facebook Prophet documentations](https://facebook.github.io/prophet/docs/quick_start.html)
2. [COVID-19 Policy Impact Evaluation: A guide to common design issues](https://arxiv.org/abs/2009.01940)
3. [Generalized logistic function](https://en.wikipedia.org/wiki/Generalised_logistic_function)
4. [Reconstructing and forecasting the COVID-19 epidemic in the United States using a 5-parameter logistic growth model](https://ghrp.biomedcentral.com/articles/10.1186/s41256-020-00152-5)
5. [Does the timing of government COVID-19 policy interventions matter? Policy analysis of an original database](https://www.medrxiv.org/content/10.1101/2020.11.13.20194761v1.full)

### Datasets:
1. `our_data/covid_metric_data/us/at_away_6_data.csv` - Contains SafeGraph mobility data - primarily extracted relevant state mobility data to use as additional regressor (indicator of policy compliance) in the Facebook Prophet model.
2. `our_data/policy_data/international/OxCGRT_latest.csv` - OxCGRT Covid Policy Tracker containing national policies for countries across the globe. Used primarily U.S state COVID policies to create holiday schedule and additional regressors (levels of policy strictness) for the Facebook Prophet model.
3. `cs156b-data/us/covid/nyt_us_counties_daily.csv` - NYT US counties daily new COVID-19 output metrics dataset. Used to create state-level COVID-19 output metrics dataframe.
4. `our_data/demographic_data/us/population_and_demographic_estimates.csv` - U.S State Level Population and Demographics dataset. Used in initial data analysis, but not in the actual model
implementation.

### Implementation Files:
1. `cs156b_plots_tk.ipynb` - Initial data analysis of policies and COVID-19 output metrics
    * Contains an initial data visualization on the effects of policy changes on the effects on the percent growth new COVID-19 cases on a state-level basis as a function of days since a policy change. Primarily focused on stay-at-home requirements and particular tightening and relaxing of
    policies.
    * Contains some correlation analysis between different policies to determine which policies
    are highly correlated to inform decisions regarding input parameters for the Facebook 
    Prophet model.
    * Contains data visualization on the number of policy changes as well as isolated
    policy changes at state-level.

2. `utils/tk_utils.py` - A compilation of all utility functions used in the implementation of the Facebook Prophet model, including:
    * Functions that deal with creating new dataframes or extracting and appending new columns to existing dataframes
    * Functions used to fit the Prophet model and predict on the forecast period
    * Plotting functions to visualize results of the trained Prophet model
    * Miscellaneous functions involving converting between datetime objects to strings and
    vice-versa
    * Optimizing functions to fit logistic growth curves and obtain optimal parameters

3. `prophet_tk.ipynb` - Actual implementation of Facebook Prophet model, both with linear
and logistic growth curve, and contains analyses of the resulting models and some experimental models. A summary of the file includes:
    * Detailed explanation of the Facebook Prophet equation
    * Counterfactual linear and logistic model trained exclusively on COVID-19 output metrics
    * Linear growth models where policy changes are modeled in the form of extended holidays to capture effects on COVID-19 output metrics
    * Linear growth models where policy changes are modeled as additional regressors
    * Adding mobility data as regressors to distinguish between repeating policies
    * Similar models as above, but with a logistic growth curve instead on smaller training periods


## ARIMA

## LSTM

## XGBoost
