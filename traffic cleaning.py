import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

# Define the file path of the CSV file
file_path = "C:/Users/mikyl/OneDrive/Documents/CSCI/Raw Data/AnnualTrafficVolume.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)
print(df)
print(df.dtypes)

# Convert 'COUNTDATE' column from int64 format to datetime format
df['COUNTDATE'] = pd.to_datetime(df['COUNTDATE'], format='%Y%m%d')
print(df.dtypes)
print(df)

# Drop unnecessary columns
df.drop(['COUNTSTATIONID', 'FormattedDate'], axis=1, inplace=True)
print(df)

# Count the number of missing values in each column
missing_count_per_column = df.isnull().sum()
print(missing_count_per_column)

# Identify columns to perform summing operation on (all columns that contain the word 'HOUR')
columns_to_sum = df.filter(like='HOUR').columns

# Group the data by 'COUNTDATE' and sum the traffic counts for each direction
df_sum = df.groupby('COUNTDATE')[columns_to_sum].sum().reset_index()

# Concatenate the summed values back to the original DataFrame
df_total = pd.concat([df, df_sum], ignore_index=True)
print(df_total)

# 'COUNTDIR' was populated with NaN for the summed values. Replace all NaN values with 'T'
df_total.replace(np.nan, 'T', inplace=True)
print(df_total)

# Sort the DataFrame by 'COUNTDATE' and reset the index
df_total = df_total.sort_values(by='COUNTDATE')
df_total = df_total.reset_index(drop=True)
print(df_total)

# Filter the DataFrame for rows where 'COUNTDIR' is either 'S' or 'P' (South or Parking direction)
directional_df = df_total[(df_total['COUNTDIR'] == 'S') | (df_total['COUNTDIR'] == 'P')]
print(directional_df)

# Remove rows where 'COUNTDIR' is 'S' or 'P' (directional filtering)
df_total = df_total[~((df_total['COUNTDIR'] == 'S') | (df_total['COUNTDIR'] == 'P'))]
print(df_total)

# Drop the 'COUNTDIR' column as it's no longer needed
df_total.drop('COUNTDIR', axis=1, inplace=True)
print(df_total)

# Create a DataFrame 'df_plot' by dropping 'COUNTDATE'
df_plot = df_total.drop('COUNTDATE', axis=1)
print(df_plot)

# Create a boxplot for the hourly traffic data
plot = sns.boxplot(data=df_plot)
plt.title('Traffic Data Distribution by Hour')
plt.ylabel('Traffic Count')
plt.xticks(rotation=45, fontsize=8)
plt.show()

# Create a new column 'DAY_OF_WEEK' based on 'COUNTDATE', where Monday = 1, Sunday = 7
df_total['DAY_OF_WEEK'] = df_total['COUNTDATE'].dt.dayofweek + 1

# Create a 'WEEKEND' column based on the 'DAY_OF_WEEK' value
df_total['WEEKEND'] = df_total['DAY_OF_WEEK'].apply(lambda x: 1 if x >= 6 else 0)

# Reset the index of the DataFrame after all transformations
df_total.reset_index(drop=True, inplace=True)
print(df_total)

# Create a DataFrame with just the columns we care about for plotting (sum of traffic and WEEKEND column)
df_comparison = df_total.copy()

# Aggregate traffic counts by weekend vs. weekday
df_comparison['TOTAL_TRAFFIC'] = df_comparison[columns_to_sum].sum(axis=1)

# Plotting boxplot to compare traffic counts on weekends vs. weekdays
plt.figure(figsize=(10, 6))
sns.boxplot(x='WEEKEND', y='TOTAL_TRAFFIC', data=df_comparison, palette="Set2")
plt.title('Weekend vs. Weekday Traffic')
plt.xlabel('')
plt.ylabel('Total Traffic Count')
plt.xticks([0, 1], ['Weekday', 'Weekend'])
plt.tight_layout()
plt.show()

# Filter the data for weekdays only (excluding weekends).
df_weekdays = df_total[df_total['WEEKEND'] == 0]

# Group by 'DAY_OF_WEEK' and calculate the mean traffic counts for each hour during weekdays
weekday_traffic_by_hour = df_weekdays.groupby(['DAY_OF_WEEK'])[columns_to_sum].mean()

# Plotting the diurnal traffic pattern during weekdays
plt.figure(figsize=(12, 6))

for i, weekday in enumerate(weekday_traffic_by_hour.index):
    sns.lineplot(data=weekday_traffic_by_hour.iloc[i, :], label=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'][i], linewidth=2)

plt.title('Weekday Diurnal Traffic Pattern')
plt.xlabel('Hour of Day')
plt.ylabel('Average Traffic Count')
plt.xlim(0, 24)
plt.xticks(ticks=np.arange(0, 24, 1), labels=range(1, 25))
plt.legend(title='Day of Week', loc='upper right')
plt.tight_layout()
plt.show()
