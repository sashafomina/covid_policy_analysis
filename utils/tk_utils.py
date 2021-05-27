# Standard plotting libraries
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import re
import seaborn as sns

# Plotly library
import plotly.express as px
import plotly.graph_objects as go

# NumPy and DateTime
import numpy as np
import datetime as dt
from datetime import timedelta

# Prophet library
from fbprophet import Prophet

# Feature importance and error metrics library
import sklearn.metrics

'''
=============================================================================================
= The below functions are utility functions used to either create or extract new dataframes =
= given the original policy dataframes given certain conditions - such as a time period,    =
= policies we want to look at change dates for, making holiday dataframes using policies,   =
= adding mobility data to a state-policy dataframe                                          =
=============================================================================================
'''

def get_df_by_dates(df, start_date=None, end_date=None):
    ''' 
    Function that takes in a dataframe with a datetime column (named 'date')
    and returns the a copy of the dataframe between the given interval
    (inclusive) 
    
    Inputs:
        df - a DataFrame containing a 'date' column
        start_date - start date of the DataFrame we want to start extraction at
        end_date - end date of the DataFrame we want to stop extraction at
        
    Returns a DataFrame containing data in the [start_date, end_date) interval.
    '''
    
    # If no end dates, just return anything past the given start_date
    if end_date == None:
        return df[(df['date'] >= start_date)].copy()
    # If no start dates, return anything before the given end_date
    elif start_date == None:
        return df[(df['date'] < end_date)].copy()
    else:
        return df[(df['date'] >= start_date) & (df['date'] < end_date)].copy().reset_index().drop(columns='index')

def make_state_policy_df(us_policy_data, us_state_daily, date_start, date_end, state=None):
    '''
    This function merges the OxCGRT policy dataframe for the US and the NYT daily COVID cases
    and deaths within the given time interval [date_start, date_end] for the given state.
    
    Inputs:
        us_policy_data - the OxCGRT policy dataset
        us_state_daily - the NYT COVID cases tracker dataset
        date_start - string representing the starting date
        date_end - string representing the end date
        state - the particular state we want to look at. If state is none, then just
                extract parts of the DataFrame within the given time window.
        
    Returns a DataFrame that combines the policy data and state COVID metric data.
    '''
    
    if state == None:
        # Load data for desired state covid metric
        temp_state_daily_df = us_state_daily.copy()
        temp_state_daily_df = temp_state_daily_df.loc[(temp_state_daily_df['date'] >= date_start) & 
                                                    (temp_state_daily_df['date'] <= date_end)]
        # Load data for desired state policy
        temp_policy_df = us_policy_data.copy()
        temp_policy_df = temp_policy_df.loc[(temp_policy_df['date'] >= date_start) & 
                                            (temp_policy_df['date'] <= date_end)]
    else:
        # Load data for desired state covid metric
        temp_state_daily_df = us_state_daily.copy()
        temp_state_daily_df = temp_state_daily_df.loc[(temp_state_daily_df['state'] == state) &
                                                 (temp_state_daily_df['date'] >= date_start) & 
                                                 (temp_state_daily_df['date'] <= date_end)]

        # Load data for desired state policy
        temp_policy_df = us_policy_data.copy()
        temp_policy_df = temp_policy_df.loc[(temp_policy_df['region_name'] == state) &
                                            (temp_policy_df['date'] >= date_start) & 
                                            (temp_policy_df['date'] <= date_end)]

    # Combine data
    temp_combined_df = temp_state_daily_df.merge(temp_policy_df, how='left',on='date')
    temp_combined_df = temp_combined_df.drop(columns=['country_code', 'region_name', 'region_code', 'jurisdiction'])
    return temp_combined_df

def make_policy_changes_df(state_policy_df, policies):
    '''
    This function extracts the columns corresponding to the policy levels
    for a given state using the state policy dataframe and creates
    a new state policy dataframe.
    
    Inputs:
        state_policy_df - the merged state COVID-19 policy and cases dataframe for a given state
        using make_state_policy_df()
        policies - the list of policies we want to retain in the new dataframe
    
    Return a policy change dataframe with desired columns.
    '''
    cols = ['date', 'state', 'cases', 'deaths'] + policies
    policy_change_df = state_policy_df[cols].copy()
    return policy_change_df

def get_number_of_policy_changes(state_policy_df, policies):
    '''
    Get the number of policy changes for a given state for a list of policies
    we want to look at
    
    Inputs
        state_policy_df - a DataFrame containing the policy information for a given state
        policies - list of policies that we want to look at
        
    Returns the resulting state policy change dataframe and the number of policy changes
    for each policy in the list of policies.
    '''
    
    # Make a copy of the state policy dataframe
    state_pchange_df = state_policy_df.copy()

    # Compute the policy changes using diff(). If diff() != 0, then we know a policy was either
    # tightened or relaxed on that given date, so set the value to 1 for that date on that policy.
    # Else, set it to 0.
    for policy in policies:  
        state_pchange_df[policy + '_diff'] = state_pchange_df[policy].diff()
        # state_pchange_df = state_pchange_df.dropna(subset=[policy + '_diff'])
        state_pchange_df[policy + '_diff'] = state_pchange_df[policy + '_diff'].apply(lambda x : 1 if x != 0 else 0)
    
    # Keeps track of the number of policy changes for each policy
    counts = []
    
    # Compute the policy changes for each policy
    for policy in policies:
        # Get total number of policy changes
        count = state_pchange_df[policy + '_diff'].sum()
        counts.append(count)
        
    return (state_pchange_df.drop(policies, axis=1).copy(), counts)

def get_isolated_policy_counts_and_dates(pchange_df, policies):
    '''
    Get the number of isolated policy changes along with the corresponding change dates for
    each desired policy.

    Input:
        pchange_df - the policy change dataframe (usually for a given state) created
            by calling get number of policy changes
        policies - a list of policies we would like to look at isolated changes for

    Return a tuple whose first entry is the number of times each policy in the list of policies was changed 
    individually without any other policies changed at the same time and whose second entry are the
    dates of isolated policy change. 
    '''
    window_df = pchange_df.copy()
    counts_no_affiliated = []
    isolated_dates = {}

    # For each policy
    for policy in policies:
        isolated_dates[policy] = []
        colname = policy + '_diff'

        # Counting number of policy changes in isolation of other changes
        count = 0
        cols = window_df.shape[1]
        for index, row in window_df.iterrows():
            if row[colname] != 0 and (row[(cols - len(policies)):].sum() - row[colname] == 0):
                # Add the current date to the date of changes corresponding to an
                # isolated policy change
                date = row['date']
                isolated_dates[policy].append(date)
                count += 1

        counts_no_affiliated.append(count)  
    return (counts_no_affiliated, isolated_dates)

def get_policy_change_dates(state_df, policy, split=False):
    '''
    Function to return all policy change dates for a given policy from the state-policy DataFrame.
    
    Input:
        state_df - the DataFrame containing both state daily cases and the policy levels
        policy - the name of the policy we want to obtain change dates for
        
    Return arrays containing the change dates for either tightening or relaxing of a policy.
    '''
    # Find out when the policy was changed
    policy_state_df = state_df.copy()
    
    # Find difference in policy levels using .diff()
    policy_state_df[policy + 'diff'] = policy_state_df[policy].diff()
    
    # Return all dates where the given policy was changed
    changes_df = policy_state_df[policy_state_df[policy + 'diff'] != 0].dropna(subset=[policy + 'diff'])
    
    if split:
        # Separate between tightening and relaxing of policies
        tightened_df = changes_df[changes_df[policy + 'diff'] > 0]
        relaxed_df = changes_df[changes_df[policy + 'diff'] < 0]
        return (np.array(tightened_df['date']), np.array(relaxed_df['date']))
    else:
        # Convert this to NumPy array
        change_dates = np.array(changes_df['date'])
        return change_dates

def find_upper_window(policy_df, max_date):
    '''
    Function to find the duration of a policy and add it as the upper window.
    
    Inputs:
        policy_df - the particular holiday policy DataFrame we want to set the upper window to
        max_date - current maximum date in the state-policy data. This accounts for the last
            policy change
            
    Returns the holiday policy dataframe with the upper window appended to a separate column.
    '''
    policy_df = policy_df.reset_index().drop(columns='index')
    policy_df = policy_df.sort_values('ds')
    policy_df['shifted_ds'] = policy_df['ds'].shift(-1)
    policy_df.iloc[-1, policy_df.columns.get_loc('shifted_ds')] = max_date
    policy_df['upper_window'] = policy_df.apply(lambda x : (x['shifted_ds'] - x['ds']).days - 1, axis=1)
    
    # To avoid overfitting, cap the upper window at 30 days
    policy_df['upper_window'] = policy_df.apply(lambda x : min(x['upper_window'], 90), axis=1)
    policy_df = policy_df.drop(columns='shifted_ds')
    return policy_df

def make_holiday_df(holiday_name, dates, lower_window, upper_window):
    '''
    Make a holiday DataFrame with the given holiday dates and the extend window (lower_window, upper_window).
    
    Inputs:
        holiday_name - name of holiday
        dates - list of datetime objects to be converted to dates in the dataframe
        [lower_window, higher_window] - window to extend holiday by
        
    Returns a DataFrame containing the holidays, their dates, and their time windows in the appropriate
    format for the Prophet model.
    '''
    holiday_df = pd.DataFrame({
        'holiday' : 'holiday_' + holiday_name,
        'ds' : pd.to_datetime(dates),
        'lower_window' : lower_window,
        'upper_window' : upper_window,
    })
    return holiday_df

def add_mobility_data_to_state_df(mobility_df, state_policy_df, state):
    '''
    Function to add mobility data into the corresponding state DataFrame for the given state
    
    Input:
        mobility_df - DataFrame containing mobility data
        state_policy_df - the corresponding state DataFrame that contains both the output metric data
            and the policy data
        state - the state we want to extract mobility data for

    Returns a copy of the original state-policy dataframe with mobility data appended to it.
    '''
    
    # Extract state mobility data
    state_mobil_data = mobility_df[mobility_df['state'] == state].copy()
    state_mobil_data = state_mobil_data.reset_index().drop(columns=['index'])
    
    # Add mobility data to the state policy df
    state_mobil_df = state_policy_df.merge(state_mobil_data, on='date')
    
    return state_mobil_df

'''
=============================================================================================
= The below functions are utility functions used to handle datetime objects and conversion  =
= between formatted datetime object strings into actual datetime objects                    = 
=============================================================================================
'''

def get_dt_from_strf(date, fstr):
    ''' 
    Function to return a datetime object from a string and its
    specified format 

    Inputs:
        date - a string representing a date in the format
        given by fstr
        fstr - the string format of the date string input
        
    Return the datetime object from the given date presented in the formatted string.
    '''
    return dt.datetime.strptime(date, fstr)

def get_period_in_dt(start_date, end_date):
    ''' 
    This function returns the period from 2 date strings
    given in the following format %Y-%m-%d
    
    Inputs:
        start_date - Start date of the dataframe we want to retrieve
        end_date - End date of the dataframe we want to retrieve
        
    Return the period between the starting date and ending date.
    '''
    dt_start = get_dt_from_strf(start_date, '%Y-%m-%d')
    dt_end = get_dt_from_strf(end_date, '%Y-%m-%d')
    delta = dt_end - dt_start
    return delta.days

def get_forecast_intervals(forecast_start, period, fstr):
    '''
    This function returns the forecast interval given a starting date (usually the
    end date of the training period) and the period of forecasting (in days). The start date
    is provided in the string format given by fstr.
    
    Inputs:
        forecast_start - string representing the start of forecast period
        period - an integer representing the length of forecasting period (in days)
        fstr - string format for the forecast_start date
        
    Return a tuple containing the start and end time of the forecast interval.
    '''
    forecast_start = get_dt_from_strf(forecast_start, fstr)
    forecast_end = forecast_start + dt.timedelta(days=period) 
    return (forecast_start, forecast_end)   

def date_to_timestep(df, min_ds=None):
    '''
    This function takes in a DataFrame with a date column 'ds', and converts all the dates into
    time steps based on the start of the time series given by min_ds.

    Inputs:
        df - a DataFrame containing the date column 'ds'
        min_ds - the minimum date we want to consider the start of the timestep series, i.e,
            date corresponding to t = 0

    Returns a copy of the original dataframe with the timestep column 't'.
    '''
    
    df_copy = df.copy()
    # If there are no mininum date provided, then we take
    # we will use the first date in the date in the dataframe
    # as the starting time series
    if min_ds == None:
        min_ds = df_copy['ds'].min()
    df_copy['t'] = df_copy['ds'] - min_ds
    df_copy['t'] = df_copy['t'].apply(lambda x : x.days)
    return df_copy

def generate_timesteps(start_date, end_date):
    '''
    Function to generate an array of timesteps from the given start date
    to the given end date.
    
    Input:
        start_date - starting date in '%Y-%m-%d' format
        end_date - similar to above, but as the end date
    
    Returns a NumPy array with the timesteps between start date and end date.
    '''
    days = get_period_in_dt(start_date, end_date)
    return np.arange(0, days)

def generate_datetimes(start_date, end_date):
    '''
    Function to generate an array of datetime objects between
    the start date and end date.
    
    Input:
        start_date - starting date in '%Y-%m-%d' format
        end_date - similar to above, but as the end date
    
    Returns a NumPy array with datetime objects between start date and end date.
    '''
    return np.arange(np.datetime64(start_date), np.datetime64(end_date), timedelta(days=1))

'''
=============================================================================================
= The below functions are utility functions used that deal with producing accuracy metrics  =
= used to evaluate Prophet models. The native Prophet library does not have error metrics   =
= built-in.                                                                                 =
=============================================================================================
'''

def get_error_values(actual_df, forecast_df, metrics='mse'):
    '''
    Function to return the desired error metrics using sklearn library
    between the actual data and the forecasted date. The actual and forecast dataframe 
    are formatted such that the columns 'ds' and 'y' / 'yhat' correspond to 
    the date value and the actual / predicted values and that the time interval line up
    with each other. This is a necessary precondition.
    
    Inputs:
        actual_df - dataframe corresponding to the actual values
        forecast_df - dataframe corresponding to the predicted values returned
            by the Prophet predictor
        metrics - error metric. Currently only support 'mse', 'rmse' 'mae' and 'r2'
        
    Return the error value for the desired metric.
    '''
    
    error = -1
    if metrics == 'mse':
        error = sklearn.metrics.mean_squared_error(actual_df['y'], forecast_df['yhat'], squared=True)
    elif metrics == 'rmse':
        error = sklearn.metrics.mean_squared_error(actual_df['y'], forecast_df['yhat'], squared=False)
    elif metrics == 'mae':
        error = sklearn.metrics.mean_absolute_error(actual_df['y'], forecast_df['yhat'])
    elif metrics == 'r2':
        error = sklearn.metrics.r2_score(actual_df['y'], forecast_df['yhat'])
    return error
    

'''
=============================================================================================
= The below functions are utility functions used produce plots using the Plotly library     =
= - namely comparing between actual output metric values, the forecasted values, and the    =
= counterfactual values                                                                     = 
=============================================================================================
'''

def plot_model_actual_predicted(input_df, fc_df, actual_df):
    '''
    Function to plot the COVID-19 output metric for the training period, the predicted values for the
    forecasting period, and the actual values for the same period. Takes in 3 dataframes: the training dataframe,
    the predicted dataframe, and the actual dataframe. Note that fc_df and actual_df must occupy the 
    same time period.
    '''
    
    # Plots the result of the data points for the training period, the predictions for forecasted period, and the actual values for
    # the forecasted period from the original data
    train_trace = go.Scatter(
        x=input_df['ds'], y=input_df['y'], 
        name='trained',
        line_color='red'
    )
    
    # Get end of training period
    train_end = input_df['ds'].max()

    # Predicted values for the forecast period
    fc_trace = go.Scatter(
        x=fc_df[fc_df['ds'] >= train_end]['ds'], y=fc_df[fc_df['ds'] >= train_end]['yhat'], 
        name='forecast',
        line_color='blue'
    )

    # Actual values for the forecast period
    actual_trace = go.Scatter(
        x=actual_df[actual_df['ds'] >= train_end]['ds'], y=actual_df[actual_df['ds'] >= train_end]['y'], 
        name='actual',
        line_color='green')


    # Confidence intervals
    upper = go.Scatter(
        x=fc_df['ds'], y=fc_df['yhat_upper'], 
        name='upper yhat',
        mode='lines', 
        line=dict(width=0.5, color='blue'),
        fill=None,
        showlegend=False
    )

    lower = go.Scatter(
        x=fc_df['ds'], y=fc_df['yhat_lower'], 
        name='lower yhat',
        mode='lines', 
        line=dict(width=0.5, color='blue'),
        fill='tonexty',
        showlegend=False
    )

    data = [train_trace, fc_trace, actual_trace, upper, lower]

    # Layout
    layout = go.Layout(
        template='seaborn',
        yaxis=dict(
            title='7 day avg. new cases'
        ),
        legend=dict(
            traceorder="reversed"
        ),
        xaxis=dict(
            title='date'
        )
    )
    
    # Show figure
    fig = go.Figure(data=data, layout=layout)
    
    # Return figure
    return fig


def plot_counterfactual_forecast_actual(actual_df, fc_df, cfc_df):
    '''
    Function to plot the COVID-19 output metric for the actual dataset, the forecasted dataset (with policies taken into
    account as either holiday effects or regressors), and the counterfactual model trained on the original dataset without
    any policy effects.
    '''
    
    # Create trace plots for the actual dataset
    actual_trace = go.Scatter(
        x=actual_df['ds'], y=actual_df['y'], 
        name='actual',
        line_color='green'
    )
    
    # Predicted values for the forecast period
    fc_trace = go.Scatter(
        x=fc_df['ds'], y=fc_df['yhat'], 
        name='forecast',
        line_color='blue'
    )

    # Actual values for the forecast period
    cfc_trace = go.Scatter(
        x=cfc_df['ds'], y=cfc_df['yhat'], 
        name='counterfactual',
        line_color='yellow')

    # Confidence intervals
    upper_fc = go.Scatter(
        x=fc_df['ds'], y=fc_df['yhat_upper'], 
        name='upper yhat',
        mode='lines', 
        line=dict(width=0.5, color='blue'),
        fill=None,
        showlegend=False
    )

    lower_fc = go.Scatter(
        x=fc_df['ds'], y=fc_df['yhat_lower'], 
        name='lower yhat',
        mode='lines', 
        line=dict(width=0.5, color='blue'),
        fill='tonexty',
        showlegend=False
    )
    
        # Confidence intervals
    upper_cfc = go.Scatter(
        x=cfc_df['ds'], y=cfc_df['yhat_upper'], 
        name='upper yhat',
        mode='lines', 
        line=dict(width=0.5, color='yellow'),
        fill=None,
        showlegend=False
    )

    lower_cfc = go.Scatter(
        x=cfc_df['ds'], y=cfc_df['yhat_lower'], 
        name='lower yhat',
        mode='lines', 
        line=dict(width=0.5, color='yellow'),
        fill='tonexty',
        showlegend=False
    )

    data = [actual_trace, fc_trace, cfc_trace, upper_fc, lower_fc, upper_cfc, lower_cfc]

    # Layout
    layout = go.Layout(
        template='seaborn',
        yaxis=dict(
            title='7 day avg. new cases'
        ),
        legend=dict(
            traceorder="reversed"
        ),
        xaxis=dict(
            title='date'
        )
    )
    
    # Show figure
    fig = go.Figure(data=data, layout=layout)
    
    # Return figure
    return fig

def plot_curve_fit(input_df, y_fitted):
    '''
    Plot the actual COVID case data given by the input dataframe versus the fitted curve values using
    the optimal parameters.

    Input:
        input_df: Input dataframe that contains at least the dates ('ds'), the timesteps corresponding to the dates
        ('t') and the actual values ('y')
        y-fitted: the fitted value obtained from using the generative function on the optimal parameters
    '''
    # Actual scatter plot of the y-values
    actual_trace = go.Scatter(
        x=input_df['ds'], y=input_df['y'], 
        name='actual',
        marker_color='red'
    )

    # Fitted y-values using the estimated coefficients
    fitted_trace = go.Scatter(
        x=input_df['ds'], y=y_fitted,
        name='fitted',
        marker_color='blue'
    )

    data = [fitted_trace, actual_trace]

    # Layout
    layout = go.Layout(
        template='seaborn',
        yaxis=dict(
            title='7 day avg. new cases'
        ),
        legend=dict(
            traceorder="reversed"
        ),
        xaxis=dict(
            title='date'
        )
    )

    # Show figure
    fig = go.Figure(data=data, layout=layout)
    fig.show()

'''
=============================================================================================
= The below functions are utility functions used to produce training / testing dataframes   =
= for the Prophet models or functions that does the actual forecasting.                     =
=============================================================================================
'''

def convert_df_to_input_df(input_df, metric_name, additional_cols=[]):
    ''' 
    Convert the state-policy dataframe to the appropriate training input dataframe.
    
    Inputs:
        input_df - the given state-policy dataframe
        metric_name - COVID-19 output metric we want to be training / forecasting on
        additional_cols - a list of additional columns to be used as regressors to the Prophet model.
            If list is empty, then no additional columns are added.
    '''
    
    # Rename the main time-series and y-value column to match the Prophet model specifications
    converted_df = input_df[['date', metric_name]].copy().rename(columns={'date' : 'ds', metric_name : 'y'})
    
    # Add the additional columns to the training dataframe
    for col in additional_cols:
        converted_df[col] = input_df[col]
        
    return converted_df.copy()

def make_preds_df(p_model, fc_period, freq='D', include_history=False, actual_df=None, colnames=[], caps_df=None):
    '''
    Make forecast dataframe to store predictions from Prophet model along with any
    additional regressors and returns the forecasted values.
    
    Inputs:
        p_model - Prophet model to make predictions from
        fc_period - an integer representing the forecasting period.
        actual_df - the actual dataframe with the regressor column names
        freq - a string of frequency type. Either 'D', 'M' or 'Y'. By default, uses daily frequency.
        include_history - boolean indicating whether we want to predict the history / training dates as well. False
            by default.
        colnames - column names of the additional regressors. Empty by default.
        caps_df - DataFrame containing the carrying capacity for from the start of the training period
            to the end of the forecast period. None by default if not logistic growth.
    '''
    # Make future dataframe for the forecast period
    future = p_model.make_future_dataframe(periods=fc_period, freq=freq, include_history=include_history)
    
    # Add additional regressor columns in to the future dataframe
    for col in colnames:
        future[col] = actual_df[col]
        
    # If we are doing logistic growth, then set the carrying capacity of
    # the future dataframe
    if caps_df is not None:
        if include_history:
            future['cap'] = caps_df['cap']
        else:
            future_caps = caps_df[caps_df['ds'] >= future['ds'].min()]['cap'].reset_index().drop(columns=['index']).copy()
            future['cap'] = future_caps
        
    fc_df = p_model.predict(future)
    
    return fc_df

def prophet_forecast(p_model, state_policy_df, start_date, end_date, fc_period, output_metric, 
                     err_metric='rmse', freq='D', colnames=[], include_history=False, caps_df=None, prior_scale=0.1):
    '''
    Function to predict the forecasting period using the given Prophet model after
    fitting it to the training dataframe.
    
    Input:
        p_model - the Prophet model we want to use to fit
        state_policy_df - the DataFrame containing state-policy data
        start_date - string representing starting training date
        end_date - string representing ending training date
        fc_period - the length of the period we want to forecase
        metric - COVID-19 metric to compare between actual and predicted values
        freq - frequency of the prediction, i.e, 'D', 'M' or 'Y'
        colnames - column names of any additional regressors
        caps_df - DataFrame containing the carrying capacity of each datapoint in the timeseries
        prior_scale - regularization factor for the additional columns. Default to 0.1.
        
    Returns the forecasting DataFrame as well the actual-predicted plot of
    the model.
    '''
    
    # Make training dataframe
    input_df = convert_df_to_input_df(
        get_df_by_dates(state_policy_df, start_date, end_date), 
        output_metric, additional_cols=colnames)
        
    # If doing logistic growth, then add carrying capacity to the input dataframe before training
    if caps_df is not None:
        input_df['cap'] = caps_df[(caps_df['ds'] >= start_date) & (caps_df['ds'] < end_date)]['cap']
        
    # Add additional regressor
    for col in colnames:
        p_model.add_regressor(col, prior_scale=prior_scale)
    
    # Fit the Prophet model to training dataset
    p_model.fit(input_df)
    
    # Get the forecasting interval given a start date and the forecasting period
    (fc_start, fc_end) = get_forecast_intervals(end_date, fc_period, '%Y-%m-%d')

    if include_history:
        # Retrieve the actual values for the forecasting period from our original state-policy dataframe
        actual_df = convert_df_to_input_df(
            get_df_by_dates(state_policy_df, start_date=start_date, end_date=fc_end), 
            output_metric, additional_cols=colnames)  
    else:
        # Retrieve the actual values for the forecasting period from our original state-policy dataframe
        actual_df = convert_df_to_input_df(
            get_df_by_dates(state_policy_df, start_date=fc_start, end_date=fc_end), 
            output_metric, additional_cols=colnames)  
    
    # Create forecasting dataframe
    fc_df = make_preds_df(
        p_model,
        fc_period,
        freq=freq,
        actual_df=actual_df,
        colnames=colnames,
        include_history=include_history,
        caps_df=caps_df
    )

    # Plot the basic model
    p = plot_model_actual_predicted(input_df, fc_df, actual_df)
    
    # Print out error metric if we specified one
    if err_metric != None:
        err = get_error_values(actual_df, fc_df, metrics=err_metric)

    return (fc_df, p, err)

'''
=============================================================================================
= The below functions are utility functions used to produce carrying capacity required by   =
= the Prophet model should we choose to model output metrics using logistic growth curves   =
=============================================================================================
'''

import scipy.optimize as optim

def logistic_func(t, a, b, K):
    '''
    Function representing a logistic growth curve 
        y(t) = c / (1 + a * e^(-bt))
    with coefficients a, b, c and time parameter t, where:
        K: the limiting value, maximum capacity for y
        b: some positive coefficient representing the incubation rate
        a: some coeffient
    The maximum growth rate t' (and its corresponding peak y(t')) is given by
        t' = ln(a) / b and y(t') = c / 2
    '''
    return K / (1 + a * np.exp(-b * t))

def gen_logistic_func(t, C_min, C_max, r, t_mid, eps):
    '''
    Function representing a generalized logistic growth curve.
    See: 
        https://ghrp.biomedcentral.com/articles/10.1186/s41256-020-00152-5
        https://en.wikipedia.org/wiki/Logistic_function
        
    The generalized logistic function (Richard's growth curve) is given by:
        y(t) = C_min + (C_max - C_min) / (1 + eps * e^(-r(t - t_mid))) ^ (1 / eps)
        
    where:
        C_min: the lower asymptote
        C_max: the upper asymptote. This is the carry capacity if C_min = 1.
        r: positive cofficient representing the daily exponential growth rate.
        1 / eps: skewness of the distribution of the cases
        t_mid: estimated tipping point
        
    which will allow us to model the declining growth of a logistic curve as opposed to
    the logistic function above.
    '''
    return C_min + (C_max - C_min) / (1 + eps * np.exp(-r * (t - t_mid)) ** (1 / eps))

def find_optimal_params(lower_bounds, upper_bounds, input_df, gen_func, theta0=None):
    '''
    Function to calculate the optimal parameters that fit the generative function to our
    input dataset using non-linear least squares estimation. Each date will be converted into 
    a timestep (t) initially.
    
    Input:
        lower_bounds - a length-n array containing the lower bounds for the estimated parameters of
            the generative function used to generate the carrying capacity for the the input dataframe, \
            where n is the number of estimated parameters for the generative function provided.
        upper_bounds - same as lower_bounds, but with upper bounds instead.
        input_df - the input DataFrame of the period with. Must at least have a date column ('ds') 
            and the output metric column ('y')
        gen_func - the generative function we are curve-fitting the data to.
        theta0 - the initial guesses for the parameters of the gen_func
            
    Returns the optimal parameters for the desired generative fucntion.
    '''
    # Convert dates to timesteps
    input_df = date_to_timestep(input_df)
    
    if theta0 == None:
        # Randomly initializes the values for the initial parameters
        theta0 = np.random.exponential(size=len(lower_bounds))
    
    # Estimate the parameters 
    popt, pcov = optim.curve_fit(gen_func, input_df['t'], input_df['y'], bounds=(lower_bounds, upper_bounds), p0=theta0)
    
    # Return the optimal params
    return input_df, popt

def gen_log_residuals(theta, ts, ys):
    '''
    Function to compute the residuals between the generalized logistic function and its parameters
    and the actual y-values. Original used for scipy.optimize.least_squares, but curve_fit is a better
    function anyways.
    
    Input:
        theta - parameters to be passed into gen_logistic_func to generate the
            y-value corresponding to the generalized logistic function
        ts - timesteps
        ys - the actual y-values corresponding to the elementwise timestep in the timesteps array 
    
    Returns the residuals between the logistic function curve and the actual y-values.
    '''
    (C_min, C_max, r, t_mid, alpha) = (theta[0], theta[1], theta[2], theta[3], theta[4])
    return gen_logistic_func(ts, C_min, C_max, r, t_mid, alpha) - ys