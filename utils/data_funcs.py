import pandas as pd
import numpy as np
import re
import math

'''
===========================================
 All functions are helpers for getting the merged 
data frame that will be used to train and test. Note 
that for now, this overall data frame only gives 
information based on the lock_down policy. 
===========================================
'''

# List of policy columns from the Oxford Policy Tracker Data
POLICIES = ['c1_school_closing', 'c2_workplace_closing',
            'c3_cancel_public_events', 'c4_restrictions_on_gatherings',
            'c5_close_public_transport', 'c6_stay_at_home_requirements',
           'c7_restrictions_on_internal_movement',
           'c8_international_travel_controls', 'e1_income_support',
           'e2_debt/contract_relief', 'e3_fiscal_measures',
           'h1_public_information_campaigns',
           'h2_testing_policy', 'h3_contact_tracing',
           'h4_emergency_investment_in_healthcare', 'h5_investment_in_vaccines',
           'h6_facial_coverings', 'h7_vaccination_policy',
           'h8_protection_of_elderly_people']

# State name to state abbreviation dictionary
STATE_TO_ABBREV ={
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

# State abbreviation to state name dictionary
abbrev_to_state = {v: k for k, v in STATE_TO_ABBREV.items()}

def snake_case_df_cols(df):
    """
    Alters df in place so that column names in snake case
    """

    # Change camel case columns to snake case, also apply lowercase
    def change_case(str):
        underscore_seen = str[0] == '_'
        lower_seen = str[0] in 'abcdefghijklmnopqrstuvwxyz'

        res = [str[0].lower()]
        for c in str[1:]:
            if c in ('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
                if not underscore_seen and lower_seen:
                    res.append('_')
                res.append(c.lower())
                lower_seen = False
            else:
                res.append(c.lower())
                lower_seen = True
            underscore_seen = c == '_'

        return ''.join(res)

    # Strip outer whitespace and replace inner whitespace with underscore
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    # Convert camelcase columns to snake case
    new_cols = []
    for index, c in df.iteritems():
        new_cols.append(change_case(index))
    df.columns = pd.Index(new_cols)

def get_county_covid_daily_data():
    us_cnty_daily_df = pd.read_csv("cs156b-data/us/covid/nyt_us_counties_daily.csv")
    us_cnty_daily_df['date'] = pd.to_datetime(us_cnty_daily_df['date'])
    return us_cnty_daily_df[['state', 'date', 'cases', 'deaths']]

def get_state_covid_daily_data():
    us_cnt_daily_df = get_county_covid_daily_data()
    us_state_daily_df = us_cnt_daily_df.groupby(['state','date']).sum()
    us_state_daily_df = us_state_daily_df.reset_index()
    not_states = ['District of Columbia', 'Guam', 'Northern Mariana Islands',
                  'Puerto Rico', 'Virgin Islands']
    us_state_daily_50 =  us_state_daily_df[~(us_state_daily_df.state.isin(not_states))]

    return us_state_daily_50

def get_state_pop_by_demographic_data():
    state_pop_df = pd.read_csv('our_data/demographic_data/us/population_and_demographic_estimates.csv')
    clean_df_cols(state_pop_df)
    state_pop_df.rename(columns={"state":"state_num", "name":"state"}, inplace=True)
    return state_pop_df

def get_state_pop_total_data():
    state_pop_df = get_state_pop_by_demographic_data()

    # Only extract sex = 0 and origin = 0 rows because these correspond to counts across sex and origin
    temp_state_pop_df = state_pop_df.loc[(state_pop_df['sex'] == 0) &
                                         (state_pop_df['origin'] == 0)]
    # Group by state and sum population
    state_total_pop_df = temp_state_pop_df.loc[:, ['state', 'popestimate2019']].groupby('state').sum('popestimate2019').copy()
    state_total_pop_df = state_total_pop_df.reset_index()

    return state_total_pop_df

def get_state_policy_data(fill=True):
    policy_df = pd.read_csv("our_data/policy_data/international/OxCGRT_latest.csv")
    policy_df['Date'] = pd.to_datetime(policy_df['Date'], format='%Y%m%d')
    snake_case_df_cols(policy_df)
    policy_df = policy_df.loc[policy_df['country_code'] == 'USA'].copy()

    if fill:
        # Replace NaNs in policy columns such as c1_school_closing.
        # At earliest NaN, fill based on previous policy value
        r1 = re.compile("[ceh][123456789]_flag")
        r2 = re.compile("[ceh][123456789]_\S")

        policy_col_names = []
        for c_index, c in policy_df.iteritems():
                if not bool(r1.search(c_index)) and bool(r2.search(c_index)):
                    policy_col_names.append(c_index)

        running_df = None
        for name_df, df in policy_df.groupby(['country_code', 'region_code']):
            df.loc[:, policy_col_names] = df[policy_col_names].ffill()
            df.fillna(0, inplace=True)
            if running_df is None:
                running_df = df.copy()
            else:
                running_df = pd.concat([running_df, df], axis = 0)

        filled_policy_data_df = running_df.copy()

        return filled_policy_data_df
    else:
        return policy_df

def get_fb_cls_data():
    fb_cls_df = pd.read_csv('our_data/misc_metric_data/us/cli_facebook_data.csv')
    fb_cls_df = fb_cls_df.replace(abbrev_to_state)
    return fb_cls_df

def get_pop_data():
    state_pop_df = pd.read_csv('cs156b-data/us/demographics/state_populations.csv')
    state_pop_df.state = state_pop_df.state.replace(abbrev_to_state)
    return state_pop_df

# getting population density info
def get_pop_density_by_state_data():
    state_area_df = pd.read_csv('our_data/physical_data/us/state_area_measurements.csv')
    state_pop_df = get_pop_data()
    state_geopop_df = state_pop_df.merge(state_area_df, on='state')
    state_geopop_df['pop_density'] = state_geopop_df.population/state_geopop_df.total_area_mi
    return state_geopop_df[['state', 'pop_density']]

def get_at_away_6_data():
    at_away_df = pd.read_csv('our_data/covid_metric_data/us/at_away_6_data.csv')
    at_away_df = at_away_df.replace(abbrev_to_state)
    at_away_df.date = pd.to_datetime(at_away_df['date'])
    at_away_df['mobile_ppl_per100'] = at_away_df.value * 100
    return at_away_df[['date', 'state', 'mobile_ppl_per100']]

def get_scaled_wages_data():
    state_median_wages_df = pd.read_csv('our_data/demographic_data/us/household_median_wages_thin.csv')
    state_median_wages_df.median_income = pd.to_numeric(state_median_wages_df['median_income'])
    state_median_wages_df['scaled_median_income'] = ((state_median_wages_df['median_income'] -
                                                      state_median_wages_df['median_income'].mean()) /
                                                      state_median_wages_df['median_income'].std())
    return state_median_wages_df[['state', 'median_income', 'scaled_median_income']]

def binary_encode_category(categories, category_name):
    '''
    Takes np array of unique categories. Returns dataframe of the categories with
    columns of their new encodings
    '''
    num_bin_digits = math.ceil(np.log2(len(categories)))
    vbinary_repr = np.vectorize(np.binary_repr)
    bin_categories = vbinary_repr(np.arange(0, len(categories)), width=num_bin_digits)

    df = pd.DataFrame({category_name: categories, 'binary': bin_categories})
    for i in range(num_bin_digits):
        df[f'{category_name}_{i}'] = [int(s) for s in df.binary.str[i]]

    df = df.drop(columns='binary')
    return df

def time_elapsed_since_policy_change(policy_df, policy_name,
                                     type_change='more_strict'):
    '''
    Takes policy_df with columns <policy_name>, 'region_name'
    where the <policy_name> column contains a series of numbers on an ordinal scale of strictness
   (highest number = most strict) from everyday in a contiguous range of days
    Returns a list of number
    of days since change in policy in a given region on a given date

    Assuming the date column has all dates in a contiguous period.
    '''
    time_elapsed_all_regions = np.zeros(policy_df.shape[0])

    for region in policy_df.region_name.unique():

        region_policies = policy_df.loc[policy_df.region_name == region, policy_name]

        time_elapsed= np.arange(0, region_policies.size)

        prev_region_policies = region_policies.shift()
        prev_region_policies.iloc[0] = 0 #assuming there was no policy in place before the first date

        if type_change == 'more_strict':
            points_of_change_mask = ((region_policies - prev_region_policies) > 0).to_numpy()
        else:
             points_of_change_mask = ((region_policies - prev_region_policies) < 0).to_numpy()


        for i in range(time_elapsed[points_of_change_mask].size):
            # point is the index at which a change has occured
            point = time_elapsed[points_of_change_mask][i]

            if i == 0 and point > 0 :
                time_elapsed[:point] = 0

            if i < time_elapsed[points_of_change_mask].size-1:
                next_point = time_elapsed[points_of_change_mask][i+1]
                time_elapsed[point:next_point] -= point
            else:
                time_elapsed[point:] -= point

        time_elapsed_all_regions[policy_df.region_name == region] = time_elapsed
        # END for; next region

    return time_elapsed_all_regions

def get_yesterdays_active_cases(case_df, window=15):
    '''
    Takes df with 'state', and (new) 'cases' column and returns a rolling window-length
    sum of new cases until today (excluding today) to get the number of active cases yesterday.
    Returns the series of yesterdays_active_cases.
    '''

    running_series = None
    for name_df, df in case_df.groupby(['state']):
        series =  df['cases'].shift().rolling(window, min_periods=1).sum()
        series.iloc[0] = 0
        if running_series is None:
            running_series = series.copy()
        else:
            running_series = pd.concat([running_series, series], axis = 0)

    return running_series.reset_index(drop=True)

def get_political_data():
    politic_df = pd.read_csv('cs156b-data/us/demographics/countypres_2000-2016.csv')
    total_votes = politic_df[politic_df.year == 2016].groupby('state').sum().candidatevotes

    politic_df = politic_df[politic_df.year == 2016].groupby(['state', 'party']).sum().reset_index()
    politic_df = politic_df[['state', 'party', 'candidatevotes']]
    politic_df.loc[politic_df.party == 'republican', 'candidatevotes'] *= -1
    politic_df = politic_df.groupby('state').sum()
    politic_df.rename(columns={'candidatevotes': 'political_index'}, inplace=True)
    politic_df.political_index =  politic_df.political_index/ total_votes.loc[politic_df.political_index.index]

    return politic_df

def get_future_cum_cases(case_df, window=15):
    '''
    Takes df with 'state', and (new) 'cases' column and returns a rolling window-length
    sum of new cases starting from today to get the number of cases over next window-length.
    Returns the series of future_cum_cases.
    '''

    running_series = None
    for name_df, df in case_df.groupby(['state']):
        series =  df['cases'].iloc[::-1].rolling(window, min_periods=1).sum().iloc[::-1]
        if running_series is None:
            running_series = series.copy()
        else:
            running_series = pd.concat([running_series, series], axis = 0)

    return running_series.reset_index(drop=True)

def get_rt_by_state():
    df =  pd.read_csv('our_data/covid_metric_data/us/rates_by_state.csv')
    rt_df = df[['Location','Rt']].copy()
    rt_df.rename(columns={'Location': 'state'}, inplace=True)
    return rt_df

def get_overall_data_df(policy="stay_at_home"):
    if policy == "stay_at_home":
        df = get_state_covid_daily_data()
        policy_df = get_state_policy_data()

        # Keep relevant columns
        policy_df = policy_df[['region_name', 'date', 'c6_stay_at_home_requirements']]

        # Add days since policy change
        policy_df.loc[:, 'days_since_more_strict'] =(
            time_elapsed_since_policy_change(policy_df, 
                                             'c6_stay_at_home_requirements',
                                             type_change='more_strict'))
        policy_df.loc[:, 'days_since_less_strict'] = (
            time_elapsed_since_policy_change(policy_df, 
                                             'c6_stay_at_home_requirements',
                                             type_change='less_strict'))

        # Rename region_name to state
        policy_df.rename(columns={'region_name': 'state'}, inplace=True)

        
        # Merge daily cases with policy df
        df = df.merge(policy_df, how='left', on=['state', 'date'])
        
        
        # Merge pop density with input df
        df = df.merge(get_pop_density_by_state_data(), on='state', how='left')

        # Merge mobility data with input df
        df = df.merge(get_at_away_6_data(), on=['date', 'state'], how='left')

        # Merge wage data with input df
        df = df.merge(get_scaled_wages_data()[['state', 'scaled_median_income']], on=['state'], how='left')

        # Merge political data with input df
        df = df.merge(get_political_data(), on=['state'], how='left')

        # Binary encode state names
        alpha_ordered_states = df.state.unique()
        alpha_ordered_states.sort()
        binary_state_encoded_df = binary_encode_category(alpha_ordered_states, 'state')
        df = df.merge(binary_state_encoded_df, on='state', how='left')
        
        # Adding day, year, month and dropping date
        df['month'] = df.date.dt.month
        df['year'] = df.date.dt.year-2020
        # Days since start of case reporting i.e. March 13, 2020
        start = pd.Period("2020-03-13", freq='H').dayofyear
        df['days_since_start'] = 365*(df.year) + df.date.dt.dayofyear - start


        return df
