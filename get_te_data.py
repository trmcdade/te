import tradingeconomics as te
import datetime
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import time
import pandas as pd
import numpy as np
import os
from os import listdir
import requests
from joblib import Parallel, delayed
from tqdm import tqdm
from functools import reduce


def chunks(l, n):
    """
    This function breaks up a list into chunks of specified 
    size: yields successive n-sized chunks from l. We use it 
    to break up the list of countries into smaller lists because 
    there is a cap on the number of characters in the list of
    countries that the api will accept for one pull.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


class TradingEconomics:

    def __init__(self,
                 today_date = datetime.now().strftime("%Y-%m-%d"),
                 existing_data = None,
                 update_date = '2019-12-16',
                 last_refresh_date = None,
                 from_scratch = None):

        """
        Author: Timothy R. McDade
        Date: 16 March 2021.
        
        This code pulls economic data from Trading Economics (TE) and outputs it at the 
        country-date level of granularity. Its output is in a wide format, where the 
        columns are country, date, and indicator (where there are as many indicators)
        as TE reports. 

        The steps to do so are as follows:
        1. Get a list of all the possible countries and indicators, and all combinations thereof. 
        2. Find the frequency at which each combination is reported. 
        3. Pull historical data for each combination. 
        4. Combine the historical data from all countries into a tall format (country, date, indicator, value)
        5. Aggregate all data reported more frequently than monthly to the monthly level. 
        6. Pivot this to a wide format (country, date, indicator1, indicator2, indicator3, ...)
        7. Sparsify the wide data: make sure it does not omit entries for which there is no data. 
        8. Backfill data reported less frequently than monthly to the previous months. 

        TODO: make it so it updates the existing file smoothly.
        """

        self.dir = 'C:/Users/trmcd/Dropbox/ML for Peace/Data/Tradingeconomics/'
        os.chdir(self.dir)
        self.other_path = "Historical Global Data/Other Files"
        self.master_path = "Historical Global Data/Master Files"

        # To log into TE, follow this process to insert your API Key.
        # First, make sure you have one at tradingeconomics.com (log into your account, 
        # go to Data --> API --> Keys). Then copy that key and put it in a txt file called 
        # apikey.txt. Put apikey.txt in the same directory as this code. 
        # It should have the format: YOURKEYHEREPART1:YOURKEYHEREPART2
        # Important: make sure there are no quotes, either single or double, 
        # around your API key in the file, and no other text except the key itself. 

        apikeyfile = open('apikey.txt', 'r')
        apikey = apikeyfile.read()
        apikeyfile.close()
        te.login(apikey)  

        self.today_date = today_date
        self.update_date = update_date
        self.cores = int(os.cpu_count())

        if from_scratch == 1:
            self.existing_data = pd.DataFrame()
            self.last_refresh_date = '2005-01-01'
        else:
            if existing_data is None:
                self.existing_data = pd.read_csv("Historical Global Data/Master Files/final_historical_te_data_2019-07-01.csv", index_col = 0)
            else:
                self.existing_data = pd.read_csv(existing_data, index_col = 0)
            if last_refresh_date is None:
                self.last_refresh_date = self.existing_data[(self.existing_data['Country'] == 'United States') & (pd.notnull(self.existing_data['currency']))]['Date'].max() + '-01'
            else:
                self.last_refresh_date = last_refresh_date

        # The list of countries for which you want to pull data. 
        self.countries = ['Albania', 'Benin', 'Colombia', 'Ecuador'
                          ,'Ethiopia', 'Georgia', 'Kenya', 'Kosovo'
                          ,'Mali', 'Mauritania', 'Morocco', 'Nigeria'
                          ,'Paraguay', 'Senegal', 'Serbia', 'Tanzania'
                          ,'Uganda', 'Ukraine', 'Zambia', 'Zimbabwe'
                          # Round 2
                          ,'Bangladesh', 'Belarus', 'Bolivia', 'Congo'
                          ,'Honduras', 'Jamaica', 'Niger', 'Philippines'
                          ,'Rwanda', 'Yemen'
                          # Round 3
                          ,'Bosnia', 'Cambodia', 'Central African Republic'
                          ,'China', 'El Salvador', 'Ghana', 'Guatemala'
                          ,'Hungary', 'India', 'Indonesia', 'Iraq', 'Jordan'
                          ,'Kazakhstan', 'Liberia', 'Libya', 'Malawi', 'Malaysia'
                          ,'Mexico', 'Mongolia', 'Mozambique', 'Myanmar', 'Nepal'
                          ,'Nicaragua', 'Pakistan', 'Paraguay', 'Philippines'
                          ,'Russia', 'South Africa', 'South Sudan', 'Thailand'
                          ]

    def get_all_country_indicators(self):
        """
        Get all the possible indicator-country combinations.
        Don't need to do this when refreshing because the list hasn't changed.
        If there's a list with today's date on it, the code wil use that list. 
        Warning: this can be a problem if you're pulling data for different
        sets of countries on the same day. In that case, you'll have to delete 
        or move the file. 
        """

        print("Get the list of all country-indicator combinations.")

        # First, pull GDP because every country has it. That way we have a
        # list of all countries.

        gdp_filename = self.other_path + '/gdp_df_' + self.today_date + '.csv'
        if len([f for f in os.listdir(self.other_path) if f.endswith('gdp_df_' + self.today_date + '.csv')]) == 0:
            gdp_df = te.getIndicatorData(indicators = 'gdp', country = 'all', output_type = 'df')
        else:
            print('Reading in country names.')
            gdp_df = pd.read_csv(gdp_filename)

        # pull the names of the countries for which we'll be pulling data
        # self.unique_country_list = list(gdp_df['Country'].unique())
        # self.countries = sorted(list(gdp_df['Country'].unique()))
        self.unique_country_list = self.countries
        countries = self.unique_country_list

        self.all_country_indicators = pd.DataFrame()
        exceptions_df = pd.DataFrame()
        filename = self.other_path + '/all_country_indicators_' + self.today_date + '.csv'
        if len([f for f in os.listdir(self.other_path) if f.endswith('all_country_indicators_' + self.today_date + '.csv')]) == 0:
            for ctry in tqdm(self.unique_country_list):
                try:
                    temp = pd.DataFrame()
                    temp = te.getIndicatorData(country = [str(ctry)], output_type = 'df')
                    self.all_country_indicators = self.all_country_indicators.append(temp)
                except requests.HTTPError as err:
                    error_temp = pd.DataFrame(columns = ['Country', 'Error Code', 'Date'])
                    error_temp['Country'] = ctry
                    error_temp['Error Code'] = err.code
                    error_temp['Date'] = dt.date.today()
                    exceptions_df = exceptions_df.append(error_temp)
                    pass
                time.sleep(1)
            self.all_country_indicators.to_csv(filename, header = True)
        else:
            print('Reading in all country indicators.')
            self.all_country_indicators = pd.read_csv(filename, index_col = 0)

        self.all_country_indicators = self.all_country_indicators[self.all_country_indicators['Country'].isin(self.countries)]

        self.unique_indicators = list(self.all_country_indicators['Category'].unique())
        self.unique_indicators.sort()

        self.hist_start = datetime.strptime('2005-01-01', '%Y-%m-%d')
        self.hist_end = datetime.strptime(self.last_refresh_date, '%Y-%m-%d')
        #turn dates into months instead of days
        # TODO: use pd.date_range() which returns datetime objects that I can change.
        # delta = relativedelta(self.hist_end, self.hist_start)
        delta = relativedelta(datetime.strptime(self.today_date, '%Y-%m-%d'), datetime.strptime(self.last_refresh_date, '%Y-%m-%d'))
        self.all_dates = set([datetime.strptime(self.last_refresh_date, '%Y-%m-%d') + relativedelta(months=x) for x in range(0, (12 * delta.years + delta.months))])

        self.all_country_indicators['Category'] =  [x.strip() for x in self.all_country_indicators['Category']]
        self.unique_combos = self.all_country_indicators[['Country', 'Category']].drop_duplicates()


    def get_indicator_frequency(self):
        """Gets how frequently each countries/indicator combination is published"""

        print("Get the frequency of the indicators.")

        n_per_chunk = 16
        self.frequency = pd.DataFrame()

        csv_file_name = self.other_path +  "frequency_combinations_" + self.today_date + ".csv"
        if len([f for f in os.listdir(self.other_path) if f.endswith("frequency_combinations_" + self.today_date + ".csv")]) == 0:
            for indicator in tqdm(self.unique_combos['Category'].unique()):
                relevant_countries_this_indicator = self.unique_combos[self.unique_combos['Category'] == indicator]['Country'].unique()
                for countries_chunk in chunks(relevant_countries_this_indicator, n_per_chunk):
                    mydata = te.getIndicatorData(country = list(countries_chunk), indicators = [indicator], output_type = 'df')
                    self.frequency = pd.concat([self.frequency, mydata], axis = 0)
                    time.sleep(1)
            self.frequency.to_csv(csv_file_name)
        else:
            print('Reading in frequency data.')
            self.frequency = pd.read_csv(csv_file_name, index_col = 0)

        self.frequency = self.frequency[['Country', 'Category', 'Frequency']]
        self.frequency['Category'] = [x.lower().strip() for x in self.frequency['Category']]


    def get_TE_data(self, initDate = None, endDate = None):
        """
        Gets several historical data series from different countries and stores
        them in several CSV files.
        The end date is always the current date.
        This is modified code from TE's github.
        """

        print("Pull the TE data.")

        if initDate is None:
            initDate = self.last_refresh_date
        if endDate is None:
            endDate = self.today_date

        countries = list(self.all_country_indicators['Country'].unique())
        # Choose your indicators. WARNING: It must be written like on https://api.tradingeconomics.com/indicators.
        self.unique_indicators = list(self.all_country_indicators['Category'].unique())
        self.unique_indicators.sort()

        # We have to use the chunks function here because the sum total of 
        # the number of country characters per API request cannot exceed 295.
        # Here is some code to check the number of characters in a chunk of 
        # your country list:
        #
        # countries = te_object.countries
        # chunked = [x for x in chunks(countries, 25)]
        # len_per_chunk = [sum([(len(a) + 1) for a in x]) for x in chunked]
        # sum([len(a) for a in countries])
        # # max num char per chunk
        # max(len_per_chunk)
        # # total API requests for running this once (note: monthly limit 
        # varies by subscription, for me it's 10k)
        # len(len_per_chunk) * len(unique_indicators)

        n_per_chunk = 19

        missing_combos = []
        for indicator in tqdm(self.unique_combos['Category'].unique()):
            relevant_countries_this_indicator = self.unique_combos[self.unique_combos['Category'] == indicator]['Country'].unique()
            for countries_chunk in chunks(relevant_countries_this_indicator, n_per_chunk):
                # find whether the files are there already. if so, skip.
                countries_this_chunk = [x for x in chunks(relevant_countries_this_indicator, n_per_chunk)]
                country_hypothetical_filenames = ["historical_data_" + self.last_refresh_date + '_til_' + self.today_date + "_" + indicator + "_" + country + ".csv" for country in countries_this_chunk]
                country_hypothetical_filenames = [item for subitem in country_hypothetical_filenames for item in subitem]
                today_pull_file_stem = "historical_data_" + self.last_refresh_date + '_til_' + self.today_date
                existing_filenames = [f for f in os.listdir(self.dir + self.other_path) if (f.startswith(today_pull_file_stem) and indicator in f)]
                missing_filenames_for_this_chunk = [x for x in country_hypothetical_filenames if x not in existing_filenames]
                # missing_combos = []
                if len(missing_filenames_for_this_chunk) > 0:
                    mydata = te.getHistoricalData(country = countries_chunk, indicator = [indicator], initDate = initDate)#, endDate = today_date) #Choose initDate or EndDate
                    if not np.all(pd.isnull(mydata)):
                        for country in countries_chunk:
                            country_indicator_file_name = self.other_path + "/historical_data_" + self.last_refresh_date + '_til_' + self.today_date + "_" + indicator + "_" + country + ".csv"
                            try:
                                if not np.all(pd.isnull(mydata[country])):
                                    for a in mydata[country][indicator][0]:
                                        if not np.all(pd.isnull(a)):
                                            dfa = pd.DataFrame(a)
                                            dfa.insert(0, 'country', country)
                                            dfa.insert(1, 'indicator', indicator)
                                dfa.to_csv(country_indicator_file_name)
                            except:
                                pass
                    else:
                        for country in countries_chunk:
                            print('Pulled data is null for ' + indicator + ' and ' + country)
                            missing_combos.append((country, indicator))
                    time.sleep(1)

        for x in missing_combos:
            # indexNames_aci = self.all_country_indicators[(self.all_country_indicators['Country'] == x[0]) & (self.all_country_indicators['Category'] == x[1])].index
            # self.all_country_indicators.drop(indexNames_aci, inplace=True)
            self.all_country_indicators = self.all_country_indicators[~((self.all_country_indicators['Country'] == x[0]) & (self.all_country_indicators['Category'] == x[1]))]
            filename = self.other_path + '/all_country_indicators_' + self.today_date + '.csv'
            self.all_country_indicators.to_csv(filename, header = True)

            # indexNames_uc = self.unique_combos[(self.unique_combos['Country'] == x[0]) & (self.unique_combos['Category'] == x[1])].index
            # self.unique_combos.drop(indexNames_uc, inplace=True)
            self.unique_combos = self.unique_combos[~((self.unique_combos['Country'] == x[0]) & (self.unique_combos['Category'] == x[1]))]


    def tall_master_file(self, initDate = None, endDate = None):
        """
        Now that we have the small files per country-indicator pair, we can
        combine them into big master files.

        TODO: make this so there are inputs of date?
        TODO: Make this so I can append new to existing?
        """

        print('Create the tall master file.')

        if initDate is None:
            initDate = self.last_refresh_date
        if endDate is None:
            endDate = self.today_date

        countries = list(self.all_country_indicators['Country'].unique())

        ## First, read in the individual data files.
        filepaths = [f'{self.other_path}/{f}' for f in listdir(self.other_path) if f.endswith('.csv') if endDate in f if initDate in f if 'historical_data_' in f]
        files = Parallel(n_jobs=self.cores)(delayed(pd.read_csv)(f) for f in tqdm(filepaths))
        new = pd.concat(files)
        tall = new
        del(files)

        ## Now format the tall file.
        tall = tall.rename(columns={'Unnamed: 0':'Date','country':'Country', 'indicator':'Category','0':'Value'})
        tall = tall[tall['Date'].notnull()] #This has the effect of dropping the 'credit rating' column, which is not associated with a date for some reason.
        tall = tall.drop_duplicates()
        tall = tall.reset_index(drop = True)
        tall['Category'] = [x.lower() for x in tall['Category']]
        tall['Date'] = [x[0:7] if isinstance(x, str) else x for x in tall['Date']]

        # There are some indicators that are reported more frequently than monthly. 
        # To arrive at a monthly number for these, we either have to:
        # 1. Sum them (e.g. jobless claims etc.),
        # 2. Average them (e.g. interest rates),  or
        # 3. Max them (e.g. coronavirus deaths to date). 
        # Below, we do just that. 

        print("Aggregate submonthly indicators.")

        initDate = self.last_refresh_date
        endDate = self.today_date

        # First: create the tall nonsparse files.
        submonthly_frequencies = list(['Daily', 'Weekly', 'Biweekly'])
        frequency = self.frequency
        submonthly_combos = frequency[frequency['Frequency'].isin(submonthly_frequencies)][['Country','Category', 'Frequency']]

        # determine heuristically which indicators to sum and which to average.
        indicators_to_max = ['coronavirus cases', 'coronavirus recovered', 'coronavirus deaths']
        indicators_to_sum = ['initial jobless claims', 'continuing jobless claims', 'foreign stock investment', 'foreign bond investment']
        indicators_to_avg = list(set(list(submonthly_combos['Category'].unique())).difference(indicators_to_sum))
        indicators_to_avg = list(set(list(submonthly_combos['Category'].unique())).difference(indicators_to_max))

        # for the indicators to max: max them at a monthly level.
        self.max_indicators_tall_nonsparse = pd.DataFrame()
        for indicator in tqdm(indicators_to_max):
            possible_countries = submonthly_combos[submonthly_combos['Category'] == indicator]
            for ctry in possible_countries['Country'].unique():
                filename = self.other_path + "/historical_data_" + self.last_refresh_date + "_til_" + self.today_date + "_" + indicator + "_" + ctry + ".csv"
                try:
                    country_df = pd.read_csv(filename)
                    country_df.rename(columns={'Unnamed: 0': 'Date', 'country':'Country', 'indicator':'Category', '0':'Value'}, inplace=True)
                    country_df['Date'] = [datetime.strftime(d, '%Y-%m') for d in pd.to_datetime(country_df['Date'])]
                    country_df['Category'] = [x.lower() for x in country_df['Category']]
                    for month in country_df['Date'].unique():
                        monthly_row = pd.DataFrame()
                        monthly_row['Country'] = country_df['Country'].unique()
                        monthly_row['Category'] = country_df['Category'].unique()
                        monthly_row['Date'] = country_df[country_df['Date'] == month]['Date'].unique()
                        monthly_row['Value'] = country_df[(country_df['Date'] == month)]['Value'].max()
                        self.max_indicators_tall_nonsparse = pd.concat([self.max_indicators_tall_nonsparse, monthly_row], axis = 0)
                except:
                    pass

        submonthly_max_filepath = self.other_path + '/max_indicators_tall_nonsparse_' + initDate + '_to_' + endDate + '.csv'
        self.max_indicators_tall_nonsparse = self.max_indicators_tall_nonsparse.reset_index(drop=True)
        self.max_indicators_tall_nonsparse.to_csv(submonthly_max_filepath)

        # for the indicators to sum: sum them at a monthly level.
        self.sum_indicators_tall_nonsparse = pd.DataFrame()
        for indicator in tqdm(indicators_to_sum):
                possible_countries = submonthly_combos[submonthly_combos['Category'] == indicator]
                for ctry in possible_countries['Country'].unique():
                    filename = self.other_path + "/historical_data_" + self.last_refresh_date + "_til_" + self.today_date + "_" + indicator + "_" + ctry + ".csv"
                    try:
                        country_df = pd.read_csv(filename)
                        country_df.rename(columns={'Unnamed: 0': 'Date', 'country':'Country', 'indicator':'Category', '0':'Value'}, inplace=True)
                        country_df['Date'] = [datetime.strftime(d, '%Y-%m') for d in pd.to_datetime(country_df['Date'])]
                        country_df['Category'] = [x.lower() for x in country_df['Category']]
                        for month in country_df['Date'].unique():
                            monthly_row = pd.DataFrame()
                            monthly_row['Country'] = country_df['Country'].unique()
                            monthly_row['Category'] = country_df['Category'].unique()
                            monthly_row['Date'] = country_df[country_df['Date'] == month]['Date'].unique()
                            monthly_row['Value'] = country_df[(country_df['Date'] == month)]['Value'].sum()
                            self.sum_indicators_tall_nonsparse = pd.concat([self.sum_indicators_tall_nonsparse, monthly_row], axis = 0)
                    except:
                        pass

        submonthly_sum_filepath = self.other_path + '/sum_indicators_tall_nonsparse_' + initDate + '_to_' + endDate + '.csv'
        self.sum_indicators_tall_nonsparse = self.sum_indicators_tall_nonsparse.reset_index(drop=True)
        self.sum_indicators_tall_nonsparse.to_csv(submonthly_sum_filepath)

        ##next, the indicators that need to be averaged.
        self.avg_indicators_tall_nonsparse = pd.DataFrame()
        for indicator in tqdm(indicators_to_avg):
                possible_countries = submonthly_combos[submonthly_combos['Category'] == indicator]
                for ctry in possible_countries['Country'].unique():
                    filename = self.other_path + "/historical_data_" + self.last_refresh_date + "_til_" + self.today_date + "_" + indicator + "_" + ctry + ".csv"
                    try:
                        country_df = pd.read_csv(filename)
                        country_df.rename(columns={'Unnamed: 0': 'Date', 'country':'Country', 'indicator':'Category', '0':'Value'}, inplace=True)
                        country_df['Date'] = [datetime.strftime(d, '%Y-%m') for d in pd.to_datetime(country_df['Date'])]
                        country_df['Category'] = [x.lower() for x in country_df['Category']]
                        for month in country_df['Date'].unique():
                            monthly_row = pd.DataFrame()
                            monthly_row['Country'] = country_df['Country'].unique()
                            monthly_row['Category'] = country_df['Category'].unique()
                            monthly_row['Date'] = country_df[country_df['Date'] == month]['Date'].unique()
                            monthly_row['Value'] = country_df[(country_df['Date'] == month)]['Value'].mean()
                            self.avg_indicators_tall_nonsparse = pd.concat([self.avg_indicators_tall_nonsparse, monthly_row], axis = 0)
                    except:
                        pass

        submonthly_avg_filepath = self.other_path + '/avg_indicators_tall_nonsparse_' + initDate + '_to_' + endDate + '.csv'
        self.avg_indicators_tall_nonsparse = self.avg_indicators_tall_nonsparse.reset_index(drop=True)
        self.avg_indicators_tall_nonsparse.to_csv(submonthly_avg_filepath)

        self.submonthly_indicators_nonsparse = pd.concat([self.max_indicators_tall_nonsparse, self.avg_indicators_tall_nonsparse, self.sum_indicators_tall_nonsparse], axis = 0).reset_index(drop = True)
        c_tuples = {(submonthly_combos['Country'].iloc[ii], submonthly_combos['Category'].iloc[ii]) for ii in range(len(submonthly_combos))}
        tall = tall.reset_index(drop=True)

        to_drop = []
        for ii in tall.index:
            _tup = (tall['Country'].loc[ii], tall['Category'].loc[ii])
            if _tup in c_tuples:
                to_drop.append(ii)

        tall = tall.drop(labels=to_drop, axis=0)
        tall = tall.append(self.submonthly_indicators_nonsparse, sort=['Country', 'Date', 'Category'])

        return tall


    def wide_master_file(self, tall = None):
        """
        Turn the tall data file into a wide format.
        """

        print('Widen the master file.')

        if tall is None:
            tall = self.master_te_data_tall

        wide = pd.pivot_table(tall,
                              index = ['Date', 'Country'],
                              columns = 'Category',
                              values = 'Value')

        ## Now format the wide file.
        new_date = [wide.index[row][0] for row in range(wide.shape[0])]
        wide.insert(loc = 0, column = 'Date', value = new_date)
        new_country = [wide.index[row][1] for row in range(wide.shape[0])]
        wide.insert(loc = 0, column = 'Country', value = new_country)
        wide.index = range(wide.shape[0])
        wide.index.name = None

        return wide


    def sparsify(self, tall = None, wide = None):
        """
        Change the format of the files so they are wide and sparse.
        """

        print('Sparsify.')

        if tall is None:
            tall = self.master_te_data_tall
        if wide is None:
            wide = self.master_te_data_wide

        tall['Category'] = [x.lower() if isinstance(x, str) else x for x in tall['Category']]

        if len(min(tall['Date'])) == 10:
            hist_start = min(tall['Date'])[:-3]
            hist_end = max(tall['Date'])[:-3]
        else:
            hist_start = min(tall['Date'])
            hist_end = max(tall['Date'])

        unique_countries = self.unique_combos['Country'].unique()
        unique_countries.sort()

        delta = relativedelta(datetime.strptime(self.today_date, '%Y-%m-%d'), datetime.strptime(self.last_refresh_date, '%Y-%m-%d'))
        all_dates = set([datetime.strptime(self.last_refresh_date, '%Y-%m-%d') + relativedelta(months=x) for x in range(0, (12 * delta.years + delta.months))])
        unique_dates = [datetime.strftime(d, '%Y-%m') for d in all_dates]
        unique_dates.sort()

        # create a new multi-index for both date and country.
        index = pd.MultiIndex.from_product([unique_countries, unique_dates],
                                           names = ['Country', 'Date'])
        all_combos = pd.DataFrame(index = index)
        new_date = [all_combos.index[row][1] for row in range(all_combos.shape[0])]
        all_combos.insert(loc = 0, column = 'Date', value = new_date)
        new_country = [all_combos.index[row][0] for row in range(all_combos.shape[0])]
        all_combos.insert(loc = 0, column = 'Country', value = new_country)
        all_combos.index = range(all_combos.shape[0])
        all_combos.index.name = None

        wide_lower = pd.merge(all_combos,
                              wide,
                              on = ['Country', 'Date'],
                              how = 'outer'
                              )
        wide_lower['Date'] = [x[0:7] if isinstance(x, str) else x for x in wide_lower['Date']]

        return wide_lower


    def bfill(self, country, category):
        """
        Backfills missing data for a country and category combination 
        to previous reporting periods,
        MAKE SURE DATE IS DATETIME OR SIMILAR AND NOT STRING!!!
        Inputted 'category' above should include caps.
        """

        self.master_te_data_wide_sparse_lower = self.master_te_data_wide_sparse_lower.reset_index(drop=True)
        cat_df = self.master_te_data_wide_sparse_lower.loc[self.master_te_data_wide_sparse_lower['Country']==country][['Date', 'Country', category]]

        cat_df = cat_df.sort_values(by=['Date'])
        cat_f = self.frequency[(self.frequency['Country']==country) & (self.frequency['Category']==category)]['Frequency'].iloc[0]

        if cat_f.lower() == 'quarterly':
            cat_df = cat_df.fillna(method = 'bfill', limit = 2)
        elif cat_f.lower() == 'yearly':
            cat_df = cat_df.fillna(method = 'bfill', limit = 11)
        elif cat_f.lower() == 'biannually':
            cat_df = cat_df.fillna(method = 'bfill', limit = 23)

        return cat_df


    def backfill_country(self, country):
        """
        creates a backfilled dataframe for given country and categories
        """

        categories = self.unique_combos.loc[self.unique_combos['Country']==country, 'Category']
        country_results = [self.bfill(country, cc.lower()) for cc in categories]
        merged_results = reduce(lambda  left,right: pd.merge(left, right, on=['Date', 'Country'], how='outer'), country_results)

        return merged_results


    def backfill(self, df = None):
        """
        Backfill data reported at a level less granular than monthly to the
        relevant previous reporting periods.
        """

        if df is None:
            df = self.master_te_data_wide_sparse_lower
        else:
            df = df

        print('Backfill the data.')
        self.results = [self.backfill_country(cc) for cc in tqdm(self.unique_combos['Country'].unique())]
        self.results = pd.concat(self.results, ignore_index=True)

        return self.results


    def main(self):
        """
        The main function that executes all the above functions in order.
        """

        print("Starting the data pull.")

        # Get all available combinations of country & indicators.
        self.get_all_country_indicators()

        # If you want, comment out self.get_indicator_frequency() to preserve hits on the API.
        # we can use a file in the folder already because this data doesn't
        # change much over time.
        self.get_indicator_frequency()

        # get the data.
        self.get_TE_data()

        # self.whats_missing() # this function isn't completely functional yet.

        ## Now: combine, widen, sparsify, and bfill the data.

        # First combine:
        self.master_te_data_tall = self.tall_master_file()
        tall_filename = self.other_path + '/master_te_data_tall_' + self.today_date + '_abbreviated.csv'
        # self.master_te_data_tall = pd.read_csv(tall_filename, index_col = 0)
        self.master_te_data_tall.to_csv(tall_filename)

        # Now widen:
        self.master_te_data_wide = self.wide_master_file()
        wide_filename = self.other_path + '/master_te_data_wide_' + self.today_date + '_abbreviated.csv'
        # self.master_te_data_wide = pd.read_csv(wide_filename, index_col = 0)
        self.master_te_data_wide.to_csv(wide_filename)

        # Now sparsify:
        self.master_te_data_wide_sparse_lower = self.sparsify(tall = self.master_te_data_tall, wide = self.master_te_data_wide)
        sparse_wide_filename = self.other_path + '/master_te_data_wide_sparse_lower_' + self.today_date + '_abbreviated.csv'
        # self.master_te_data_wide_sparse_lower = pd.read_csv(sparse_wide_filename, index_col = 0)
        self.master_te_data_wide_sparse_lower.to_csv(sparse_wide_filename)

        # Now backfill.
        # The combined data will be written to a master directory.
        final = self.backfill(df = self.master_te_data_wide_sparse_lower)
        final = final[final['Date'] <= self.today_date[:7]]
        final_filename = self.master_path + '/final_historical_te_data_' + self.today_date + '_abbreviated.csv'
        final.to_csv(final_filename)

        # Run tests.
        # test()

        print("All done.")


if __name__ == "__main__":

    te_object = TradingEconomics(today_date = datetime.now().strftime("%Y-%m-%d"),
                                update_date = '2005-01-01',
                                last_refresh_date = '2005-01-01',
                                from_scratch = 1)

    te_object.main()