import os
import datetime

# All Functions

# # Function to check if a string represents a valid integer within a specified range
def is_valid_integer(value, min_val, max_val):
    try:
        num = int(value)
        if num < min_val or num > max_val:
            return False
        return True
    except ValueError:
        return False

# Function to check if a file path exists and has a .csv extension
def is_valid_csv_file(file_path):
    print("Checking file path:", file_path)
    if not os.path.isfile(file_path):
        print("File does not exist, or full path was not provided (be sure to include folder directory).")
        return False
    if not file_path.lower().endswith('.csv'):
        print("File does not have a .csv extension.")
        return False
    else: 
        print(f"Valid file path for {file_path}.")
        return True
    
# Custom function to adjust year_woy column
def adjust_year_woy(row):
    if row["week_of_year"] == 53:
        if row["month"] == 12:  # December date
            return row["year"] * 100 + row["week_of_year"]
        elif row["month"] == 1:  # January date
            return str((row["year"] - 1)) + '_' + str(row["week_of_year"])
    else:
        return str(row["year"]) + '_' + str(row["week_of_year"])
    
# function to determine what month a week falls into if it spans two months
# in this case, evaluating Thursday as the middle point of Mon-Sun week
# whatever Thursday is, is the month assignment, since it is the tiebreaker
def get_month_for_week(year, week_number):
    mid_week_date = datetime.datetime.strptime(f'{year}-W{week_number}-4', "%Y-W%W-%w").date()
    return mid_week_date.month


def get_week_dates_str(year, week):
    # Check if week 53 exists for the given year
    if datetime.date(year, 12, 28).isocalendar()[1] == 53:
        # Get the first day of the week using ISO year, ISO week, and day of the week (Monday)
            start_of_week = datetime.datetime.fromisocalendar(year, week, 1)
            
            # Calculate the last day of the week
            end_of_week = start_of_week + datetime.timedelta(days=6)
            
            # Format the dates as strings
            start_date_str = start_of_week.strftime('%Y-%m-%d')
            end_date_str = end_of_week.strftime('%Y-%m-%d')
            
            return f"{start_date_str} - {end_date_str}"
    else:
        if week != 53:
            # Get the first day of the week using ISO year, ISO week, and day of the week (Monday)
            start_of_week = datetime.datetime.fromisocalendar(year, week, 1)
            
            # Calculate the last day of the week
            end_of_week = start_of_week + datetime.timedelta(days=6)
            
            # Format the dates as strings
            start_date_str = start_of_week.strftime('%Y-%m-%d')
            end_date_str = end_of_week.strftime('%Y-%m-%d')
            
            return f"{start_date_str} - {end_date_str}"