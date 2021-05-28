# CS 156b - Analysis of Policy Effectiveness and Time-Sensitivity of Policy Implementation on COVID-19 Output Metrics

## disco_panda Team Members

Thai Khong - Did some initial data analysis regarding effects of policy changes on COVID-19 output metrics. Focused on analyzing policy effectiveness using the Facebook Prophet models (with linear and logistic growth curve)

Sasha Fomina -

Beau Lobodin - Did some initial data analysis regarding effects of policy changes on COVID-19 output metrics. Focused on analyzing policy effectiveness via recursive multi-step forecasting with XGBoost and Encoder/Decoder LSTMs

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
and logistic growth curve, and contains analyses and visualizations of the resulting models and some other experimental models. A summary of the file includes:
    * Detailed explanation of the Facebook Prophet equation
    * Counterfactual linear and logistic model trained exclusively on COVID-19 output metrics
    * Linear growth models where policy changes are modeled in the form of extended holidays to capture effects on COVID-19 output metrics
    * Linear growth models where policy changes are modeled as additional regressors
    * Adding mobility data as regressors to distinguish between repeating policies
    * Similar models as above, but with a logistic growth curve instead on smaller training periods


## ARIMA

## LSTM
### References:
1. [A Novel Intervention Recurrent autoencoder for real time forecasting and non-pharmaceutical intervention selection to curb the spread of Covid-19 in the world] (https://www.medrxiv.org/content/10.1101/2020.05.05.20091827v2.full)
2. [Forecasting Treatment Responses Over Time Using Recurrent Marginal Structural Networks] (https://vanderschaar-lab.com/papers/nips_rmsn.pdf)
3. [Using keras for designing encoder/decoder LSTM models] (https://betterprogramming.pub/a-guide-on-the-encoder-decoder-model-and-the-attention-mechanism-401c836e2cdb)
4. [Background information on the encoder/decoder method] (https://towardsdatascience.com/understanding-encoder-decoder-sequence-to-sequence-model-679e04af4346)
5. [Keras Neural Net Documentation] (https://keras.io/api/)

### Datasets:
1. `our_data/covid_metric_data/us/at_away_6_data.csv` - Contains SafeGraph mobility data - used to create daily feature value of mobile_ppl_per100
2. `our_data/policy_data/international/OxCGRT_latest.csv` - OxCGRT Covid Policy Tracker containing national policies for countries across the globe. Used only U.S state COVID policies and their strictness
levels as daily features in the model.
3. `cs156b-data/us/covid/nyt_us_counties_daily.csv` - NYT US counties daily new COVID-19 output metrics dataset. Used to create state-level COVID-19 output metrics dataframe.
4. `our_data/physical_data/us/state_area_measurements.csv` - Used along with state_population data to compute population density per state, which was an input feature for the model.
5. `cs156b-data/us/demographics/state_populations.csv` - Used along with state_area data to compute population density per state, which was an input feature for the model.
6. `our_data/demographic_data/us/household_median_wages_thin.csv`- Median wage data from the US census. Used to compute a scaled median wage feature for each state
7. `cs156b-data/us/demographics/countypres_2000-2016.csv` - Presidential vote counts for different parties. Used to compute a scaled political index feature for each state.

### Implementation Files:
1. `initial_data_manip.ipynb` - A notebook for doing some initial data visualizations of covid metrics with respect to policy changes. This notebook includes:
    * Functions for plotting covid daily case counts and policy changes for US states over specific periods of time
    * Functions for plotting the frequency of policy changes across all US states
    * Code for evaluating the correlation between specific policies and their strictness changes
    * Code for looking at raw dataframes and transforming them into different data representations

2. `utils/data_funcs.py` - A compilation of all data utility functions used by the Encoder/Decoder LSTM model. These functions primarily deal with converting raw csv files into usable dataframes,
and converting data into a format suitable for input features.

3. `lstm.ipynb` - An implementation of an Encoder/Decoder LSTM model for recursive multi-step forecasting of log daily covid cases with respect to school closures. The python notebook contains:
  * Functions responsible for creating datasets usable by the LSTM for training, testing, and simulating
  * A Keras model subclass for the encoder and decoder models
  * Functions for training and evaluating the performance of the custom encoder and decoder LSTM models
  * Functions for running counterfactual policy simulations using the encoder and decoder LSTM models

## XGBoost
### References:
1. [XGBoost Documentation] (https://xgboost.readthedocs.io/en/latest/)
2. [SHAP Library Documentation for Feature Interpretation] (https://shap.readthedocs.io/en/latest/index.html)
3. [Guide to interpreting XGBoost models] (https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27)

### Datasets:
1. `our_data/covid_metric_data/us/at_away_6_data.csv` - Contains SafeGraph mobility data - used to create daily feature value of mobile_ppl_per100
2. `our_data/policy_data/international/OxCGRT_latest.csv` - OxCGRT Covid Policy Tracker containing national policies for countries across the globe. Used only U.S state COVID policies and their strictness
levels as daily features in the model.
3. `cs156b-data/us/covid/nyt_us_counties_daily.csv` - NYT US counties daily new COVID-19 output metrics dataset. Used to create state-level COVID-19 output metrics dataframe.
4. `our_data/physical_data/us/state_area_measurements.csv` - Used along with state_population data to compute population density per state, which was an input feature for the model.
5. `cs156b-data/us/demographics/state_populations.csv` - Used along with state_area data to compute population density per state, which was an input feature for the model.
6. `our_data/demographic_data/us/household_median_wages_thin.csv`- Median wage data from the US census. Used to compute a scaled median wage feature for each state
7. `cs156b-data/us/demographics/countypres_2000-2016.csv` - Presidential vote counts for different parties. Used to compute a scaled political index feature for each state.

### Implementation Files:
1. `initial_data_manip.ipynb` - A notebook for doing some initial data visualizations of covid metrics with respect to policy changes. This notebook includes:
    * Functions for plotting covid daily case counts and policy changes for US states over specific periods of time
    * Functions for plotting the frequency of policy changes across all US states
    * Code for evaluating the correlation between specific policies and their strictness changes
    * Code for looking at raw dataframes and transforming them into different data representations

2. `utils/data_funcs.py` - A compilation of all data utility functions used by the XGboost models. These functions primarily deal with converting raw csv files into usable dataframes,
and converting data into a format suitable for input features.

2. `xgboost_closure.ipynb` - An implementation of three XGBoost models for recursive multi-step forecasting of daily covid cases with respect to school closures. The python notebook contains:
    * Code for compiling the input dataframes for the LSTM models
    * Functions for training and testing the XGBoost Models
    * Functions for running counterfactual policy simulations using the XGboost models
