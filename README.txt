README: TradingEconomics Output Data File Structure and Script Overview
Author: Tim McDade
Date: 16 March 2021

This folder contains all the working and final products of the TradingEconomics work stream.

The script "get_te_data.py" is designed to download all economic data TradingEconomics holds for all countries 
(or selected countries) and output it in a csv file where the rows are unique country-month combinations and 
the columns are all the economic indicators that TradingEconomics holds. The temporal scope can be adjusted, 
but the earliest data we've retrieved so far is from January 2005. The script's output files are contined in 
the folder "\Historical Global Data".

The TradingEconomics API outputs one csv per country-indicator combination, where the rows are unique data points 
(be they reported submonthly, monthly, quarterly, yearly, etc.) and the columns are date, country, indicator name, 
and value of the indicator. All these csvs are saved in the folder "\Historical Global Data\Other Files". 
The script then combines all files pulled into one tall file, called "master_te_data_tall_yyyy-mm-dd_abbreviated" 
and saves it into the same folder. It should be noted that this step of the process entails checking the frequency 
at which each country-indicator combintion is reported: Belgium and China could report their unemployment numbers 
monthly and semi-anually, respectively. This step also aggregates indicators reported submonthly to a monthly level 
(for some indicators like jobless reports, summing is appropriate, for others like interest rates averaging is 
appropriate; for coronavirus cases to date, max is appropriate). The submonthly frequencies are "daily", "weekly", 
and "biweekly". I had to determine heuristically the indictors to sum, average, and max. The ones to sum are 
['initial jobless claims', 'continuing jobless claims', 'foreign stock investment', 'foreign bond investment']; the 
ones to max are ['coronavirus cases', 'coronavirus recovered', 'coronavirus deaths'], and all others are averaged 
(there are about 50 others). 

The next step is turning that tall file, which has (#countries * #reportingdates * #indicators) rows and four 
columns, into a wide file where there are (#countries * #reportingdates) rows and (#indicators, usually in the 
hundreds) columns. This is saved as "master_te_data_wide_yyyy-mm-dd_abbreviated" in the Other Files folder. 

The next step is to take that wide file and ensure that it is sparse: for every country, each date should be listed,
even if the country did not report any data on that date. This file is called 
"master_te_data_wide_sparse_lower_yyyy-mm-dd_abbreviated" and is also saved in the Other Files folder. 

The next step is to backfill all the missing (sparse) data with the next reported data point. For example, if a 
country reports unemployment quarterly, then the unemployment numbers must be backfilled into the prior two months.
I'm backfilling based on the frequency of reporting. For example, if the frequency is quarterly, I'm backfilling 
two months (fill the empty months, don't overwrite the previous reporting period); if the frequency is yearly, 
I'm backfilling 11 months; if it's biannually, I'm backfilling 23 months. This is the final output and is 
called "final_historical_te_data_yyyy-mm-dd_abbreviated". It is stored in the Master Files folder.

A few nomenclature notes:
1. "yyyy-mm-dd" refers to the date the report is pulled. 
2. "abbreviated" in filenames means that the report does not contain every country. This is useful as we roll out
   our forecast to different countries. 

Future work: 

I hope to streamline the code's ability to update existing files, rather than pulling new data each time. 