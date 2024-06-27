from helpers import is_valid_integer, is_valid_csv_file, adjust_year_woy, get_month_for_week, get_week_dates_str
import sys
import numpy as np
import pandas as pd
import warnings
import datetime
from tabulate import tabulate
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")



# Check if the correct number of arguments is provided
if len(sys.argv) != 3:
    print("Usage: python staff_forecast.py <folder/csv_file> <num_staff>")
    sys.exit(1)

# Save command line arguments to variables
csv_file = sys.argv[1]
num_staff = int(sys.argv[2])

# Validate the CSV file path
if not is_valid_csv_file(csv_file):
    print("Error: The first argument must be a valid .csv file path.")
    sys.exit(1)

# Validate the number of staff
if not is_valid_integer(num_staff, 12, 1200):
    print("Error: The second argument must be an integer between 12 and 1200.")
    sys.exit(1)

# Verify for user the entered arguments
print("\n")
print("CSV file:", csv_file)
print("Number of full-time/year-round staff members:", num_staff)
print("\n")

# read csv file into pandas dataframe
try:
    df = pd.read_csv(csv_file)
except pd.errors.EmptyDataError:
    print("Error: The CSV file is empty.")
    sys.exit(1)

# check if required columns are present
required_columns = ['LandingSite', 'Date_TimeCaught']
if not set(required_columns).issubset(df.columns):
    print("Error: The CSV file must contain the following columns:", required_columns)
    sys.exit(1)
    
# pop off the LandingSite prefix
df['LandingSite'] = df['LandingSite'].str.replace('LandingSite_CaptureSite', '')
# keep only required columns
df = df[required_columns]

# Create Time features from Date_TimeCaught 
df["Date_TimeCaught"] = pd.to_datetime(df["Date_TimeCaught"])
df["year"] = df["Date_TimeCaught"].dt.year
df["month"] = df["Date_TimeCaught"].dt.month
df["week_of_year"] = df["Date_TimeCaught"].dt.isocalendar().week

# Apply function to create the year_woy column
df["year_woy"] = df.apply(adjust_year_woy, axis=1)

# Take only data from 2014 onward for the training data -- no matter what
# the prior years were too preliminary while relationship building between conservationists and fishers
df_2014up = df[df['year'] >= 2014]

# Find the most recent date in df_train
most_recent_date = df_2014up['Date_TimeCaught'].max()
# Calculate the date one year prior
one_year_ago = most_recent_date - datetime.timedelta(days=365)

# Separate data from the last year into df_test
df_test = df_2014up[df_2014up['Date_TimeCaught'] >= one_year_ago]
# Remove data corresponding to df_test from df_train
df_train = df_2014up[df_2014up['Date_TimeCaught'] < one_year_ago]

# After running test, we will use all of the data to make prediction for next year
df_final = df_2014up

# Display summary
print("Most recent date in dataset:", most_recent_date)
print("One year prior:", one_year_ago)
print("Number of rows in train data after separation:", len(df_train))
print("Number of rows in test data (the past year):", len(df_test))
print("\n")

# Apply the function to each row in the test DataFrame to create the 'month_round' column for cusping week clarification
# this is necessary for ALL further mean by month comparisons
df_test['month_round'] = df_test.apply(lambda row: get_month_for_week(row['year'], row['week_of_year']), axis=1)

# training data for week comparison: group, then count the number of turtle captures
turtle_week_train = df_train.groupby(['LandingSite', 'year', 'week_of_year', 'year_woy']).size().reset_index(name='number_turtles')
# training data for month comparison: group, then count the number of turtle captures
turtle_month_train = df_train.groupby(['LandingSite', 'year', 'month', 'week_of_year', 'year_woy']).size().reset_index(name='number_turtles')
# testing data for week comparison: group, then count the number of turtle captures
turtle_week_test = df_test.groupby(['LandingSite', 'year', 'week_of_year', 'year_woy']).size().reset_index(name='number_turtles')
# testing data for month comparison: group, then count the number of turtle captures
turtle_month_test = df_test.groupby(['LandingSite', 'year', 'month_round', 'week_of_year', 'year_woy']).size().reset_index(name='number_turtles')
# training data for week comparison: group, then count the number of turtle captures

# here is the set for the final prediction
turtle_week_final = df_final.groupby(['LandingSite', 'year', 'week_of_year', 'year_woy']).size().reset_index(name='number_turtles')
# training data for month comparison: group, then count the number of turtle captures
turtle_month_final = df_final.groupby(['LandingSite', 'year', 'month', 'week_of_year', 'year_woy']).size().reset_index(name='number_turtles')


# Calculating the means of the training data by week and site
mean_train = turtle_week_train.groupby(['LandingSite', 'week_of_year'])['number_turtles'].mean().reset_index(name='mean_turtles')
mean_train['mean_turtles'] = mean_train['mean_turtles'].round()
# Calculating the means of the training data by month and site 
month_mean_train = turtle_month_train.groupby(['LandingSite', 'month'])['number_turtles'].mean().reset_index(name='mean_turtles')
month_mean_train['mean_turtles'] = month_mean_train['mean_turtles'].round()

# Calculating the means of the final data by week and site
mean_final = turtle_week_final.groupby(['LandingSite', 'week_of_year'])['number_turtles'].mean().reset_index(name='mean_final')
mean_final['mean_final'] = mean_final['mean_final'].round()
# Calculating the means of the final data by month and site 
month_mean_final = turtle_month_final.groupby(['LandingSite', 'month'])['number_turtles'].mean().reset_index(name='mean_final')
month_mean_final['mean_final'] = month_mean_final['mean_final'].round()


# Preparing the test data with rounded months to compare to the mean by month
# creating template to ensure any weeks not represented in the data are accounted for
# these additional weeks need to have their NaNs replaced with a count of 0
# additionally, given the year and week of the new rows added, a month needs to be assigned 
one_year_ago_year = one_year_ago.year
one_year_ago_week = one_year_ago.isocalendar()[1]

# Create a DataFrame with subsequent 53 weeks for each site, beginning with one year before dataset ends
template_data = []
for site in turtle_month_test['LandingSite'].unique():
    for week in range(one_year_ago_week, one_year_ago_week + 54):
        # Adjust year if the week spills over into the next year
        year = one_year_ago_year + (week // 53)
        # Calculate week of the year within the current year
        week_of_year = week % 53 if week % 53 != 0 else 53
        
        template_data.append((site, year, week_of_year))

# Create template DataFrame
template_df = pd.DataFrame(template_data, columns=['LandingSite', 'year', 'week_of_year'])
# Merge the template DataFrame with turtle_month_test DataFrame to fill in missing time values
month_round = pd.merge(template_df, turtle_month_test[['LandingSite', 'year', 'week_of_year', 'number_turtles']], 
                       on=['LandingSite', 'year', 'week_of_year'], how='left')

# Fill missing values with 0 for number_turtles column
month_round['number_turtles'].fillna(0, inplace=True)
# Calculate the month_round column using the function defined earlier
month_round['month_round'] = month_round.apply(lambda row: get_month_for_week(row['year'], row['week_of_year']), axis=1)
# Reorder columns and sort
month_round_test = month_round[['LandingSite', 'year', 'month_round', 'week_of_year', 'number_turtles']]
month_round_test = month_round_test.sort_values(by=['LandingSite', 'year', 'week_of_year'])

tt_mean_month = pd.merge(month_round_test, month_mean_train[['LandingSite', 'month', 'mean_turtles']], 
                            left_on=['LandingSite', 'month_round'], right_on=['LandingSite', 'month'], how='left')
tt_mean_month['mean_turtles'].fillna(0, inplace=True)

# Rename the column if needed
tt_mean_month.rename(columns={'mean_turtles': 'mean_turtles_month'}, inplace=True)

# Sort the DataFrame first by 'week_of_year' and then by 'LandingSite'
tt_mean_month_sorted = tt_mean_month.sort_values(by=['week_of_year', 'LandingSite'])
tt_mean_month = tt_mean_month_sorted[['week_of_year', 'LandingSite', 'month_round', 'mean_turtles_month', 'number_turtles']]
tt_mean_month = tt_mean_month.merge(month_mean_final, how='left', left_on=['LandingSite', 'month_round'], right_on=['LandingSite', 'month']).fillna(0)


# Preparing the test data to compare to the mean by month
# creating template to ensure any weeks not represented in the data are accounted for
# these additional weeks need to have their NaNs replaced with a count of 0
all_weeks = pd.DataFrame({'week_of_year': range(1, 54)})
all_sites = pd.DataFrame({'LandingSite': turtle_week_train['LandingSite'].unique()})

# Cartesian product to get all possible combinations of 'week_of_year' and 'LandingSite'
all_combinations = all_weeks.assign(key=1).merge(all_sites.assign(key=1), on='key').drop(columns='key')

# Merge with the aggregated DataFrame to fill in missing values with 0
train_mean_week = all_combinations.merge(mean_train, how='left', on=['LandingSite', 'week_of_year']).fillna(0)
tt_mean_week = train_mean_week.merge(turtle_week_test, how='left', on=['LandingSite', 'week_of_year']).fillna(0)
tt_mean_week = tt_mean_week.merge(mean_final, how='left', on=['LandingSite', 'week_of_year']).fillna(0)

tt_mean_week.drop('year', axis=1, inplace=True)
tt_mean_week.drop('year_woy', axis=1, inplace=True)


tt_all = pd.merge(tt_mean_week, tt_mean_month[['week_of_year', 'LandingSite', 'mean_turtles_month', 'mean_final']], 
                         on=['week_of_year', 'LandingSite'], how='left')

# if time, come back to fix this bug below, i don't know why week 52 was such a bother
tt_all.drop_duplicates(inplace=True)
rows_to_drop = tt_all[(tt_all['week_of_year'] == 52) & (tt_all['number_turtles'] == 1)]
tt_all.drop(rows_to_drop.index, inplace=True)

tt_all['mean_turtles_both'] = (tt_all['mean_turtles_month'] + tt_all['mean_turtles']) / 2
tt_all['mean_turtles_both'] = tt_all['mean_turtles_both'].round()

tt_all['mean_final_both'] = (tt_all['mean_final_x'] + tt_all['mean_final_y']) / 2
tt_all['mean_final_both'] = tt_all['mean_final_both'].round()

# Predictions for next year are now tt_all['mean_final_both']

# Calculate RMSE
rmse_both = np.sqrt(mean_squared_error(tt_all['number_turtles'], tt_all['mean_turtles_both']))
rmse_both = rmse_both.round(3)
# Calculate R-squared
r_squared_both = r2_score(tt_all['number_turtles'], tt_all['mean_turtles_both']).round(3)

print("Evaluation of Best Model:")
print(f"RMSE: {rmse_both} turtles")
print(f"R-squared: {r_squared_both}")



# prep for template dataframe for forecast 
today_year = most_recent_date.year
today_week = most_recent_date.isocalendar()[1]
template_forecast = []
for site in df_final['LandingSite'].unique():
    for week in range(today_week, today_week + 54):
        # Adjust year if the week spills over into the next year
        year = today_year + (week // 54)
        # Calculate week of the year within the current year
        week_of_year = week % 53 if week % 53 != 0 else 53
        
        template_forecast.append((site, year, week_of_year))

# Create template DataFrame for forecast
template_forecast = pd.DataFrame(template_forecast, columns=['LandingSite', 'year', 'week_of_year'])
# Fill template for total forecast
forecast = pd.merge(template_forecast, tt_all[['LandingSite', 'week_of_year', 'mean_final_both']], 
                       on=['LandingSite', 'week_of_year'], how='left')
forecast = forecast.reindex(columns=['year', 'week_of_year', 'LandingSite', 'mean_final_both'])
forecast = forecast.sort_values(by=['year', 'week_of_year', 'LandingSite'])



# apply own function to create the new column for week dates for user legibility
forecast['week_dates'] = forecast.apply(lambda row: get_week_dates_str(row['year'], row['week_of_year']), axis=1)
# more legibility 
new_df = forecast[['week_dates', 'LandingSite', 'mean_final_both']]
new_df = new_df.reset_index(drop=True)
new_df = new_df.rename(columns={'week_dates': 'Week Dates (YYYY-MM-DD)', 
                                 'LandingSite': 'Landing Site', 
                                 'mean_final_both': 'Number of Turtles'})
# converting 'Number of Turtles' column to integer type
new_df['Number of Turtles'] = new_df['Number of Turtles'].astype(int)

# creating a series to store the total number of turtles by week across sites, in same indexing as original
total_turtles_per_week = new_df.groupby('Week Dates (YYYY-MM-DD)')['Number of Turtles'].transform('sum')
# finding proportions of site's turtles to total turtles across sites for staff calculations
new_df['Proportion'] = new_df['Number of Turtles'] / total_turtles_per_week
new_df['Proportion'] = new_df['Proportion'].round(3)


# Filter out rows with missing or infinite values in the 'Proportion' column
new_df = new_df[new_df['Proportion'].notna() & np.isfinite(new_df['Proportion'])]


# Calculate the remaining staff available for allocation (taking one away per site as a minimum)
remaining_staff = num_staff - len(new_df['Landing Site'].unique())
# Apply proportion to distribute the remaining staff among the sites for each week
new_df['Allocated Staff'] = (remaining_staff * new_df['Proportion']).astype(int)
# Ensure that the calculated staff numbers are whole numbers and do not exceed the total number of staff members
new_df['Allocated Staff'] = new_df['Allocated Staff'].clip(upper=num_staff - 1) + 1

# find third quartile value to choose threshold for needing extra help (especially oct - jan)
seventy_fifth_percentile = total_turtles_per_week.quantile(0.75)
# create the boolean 'Need_Temps' column based on the condition using prior series with weekly totals
new_df['Need_Temps'] = total_turtles_per_week > seventy_fifth_percentile


# replace consecutive duplicate values with an empty string (so it looks blank)
new_df['Week Dates (YYYY-MM-DD)'] =new_df['Week Dates (YYYY-MM-DD)'].mask(new_df['Week Dates (YYYY-MM-DD)'].duplicated(), '')

# number of unique 'Landing Site' values
num_unique_sites = len(new_df['Landing Site'].unique())
# initialize a list to store tabulated rows
tabulated_rows = []
# the column names
columns = new_df.columns.tolist()
# iterate over the rows of the DataFrame
for i, row in enumerate(new_df.itertuples(index=False), start=1):
    tabulated_rows.append(row)
    # add a horizontal line after every 'n' rows (in case number of Landing Sites changes)
    if i % num_unique_sites == 0 and i != len(new_df):
        tabulated_rows.append(['---' for _ in range(len(row))])  # Add a horizontal line


# Display
print(f"\nThreshold of Turtles per Week before Temporary Workers/Interns are Needed: {seventy_fifth_percentile}")
print(tabulate(tabulated_rows, headers=columns, tablefmt='pretty', showindex='never'))
# clarify explicit threshold of needing temp workers for user
# print(f"Threshold of Turtles per Week before Temporary Workers/Interns are Needed: {seventy_fifth_percentile}")
