import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

file_path = "C:/Users/mikyl/OneDrive/Documents/CSCI/Raw Data/aqnow.csv"

# Read the CSV file
df = pd.read_csv(file_path)
print(df)

# Filter out any rows that exactly match the column names
df = df[~df.apply(lambda row: row.isin(df.columns).all(), axis=1)]

# Reset the index after removing the rows
df = df.reset_index(drop=True)

# Drop columns that are the same for all data points
df.drop(['HourObserved', 'LocalTimeZone', 'ReportingArea', 'StateCode', 'Latitude', 'Longitude', 'CategoryNumber'],
        axis=1, inplace=True)

# Set display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Convert 'AQI' column to numeric, coercing errors into NaN
df['AQI'] = pd.to_numeric(df['AQI'], errors='coerce')

# Verify the data type
print(df['AQI'].dtype)

# Count the number of missing values in each column
missing_count_per_column = df.isnull().sum()
print(missing_count_per_column)

# Count the total number of missing values in the DataFrame (none and NaN)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
total_missing = df.isnull().sum().sum()
print(total_missing)

# Vis for 'CategoryName'
# Count the frequency of each category in the 'CategoryName' column
category_counts = df['CategoryName'].value_counts()

# Plot the frequency of each category
category_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Frequency of Air Quality Index Categories')
plt.xticks(rotation=45)
plt.xlabel('')
plt.tight_layout()
plt.show()

# Vis for 'ParameterName'
pollutant_list = df['ParameterName'].unique()

plt.figure(figsize=(10, 6))
sns.boxplot(x='ParameterName', y='AQI', data=df, order=pollutant_list, palette="Set2")
plt.title('AQI Distribution by Pollutant')
plt.xlabel('Pollutant')
plt.ylabel('Air Quality Index (AQI)')
plt.tight_layout()
plt.show()

# Vis for numeric 'AQI'
df['AQI'].hist(bins=np.arange(df['AQI'].min(), df['AQI'].max(), step=5))
plt.title('Distribution of Air Quality Index')
plt.xlabel('Air Quality Index')
plt.xlim(0)
plt.ylabel('Frequency')
plt.grid(False)
plt.tight_layout()
plt.show()

print(df['AQI'].describe())

# Save the cleaned DataFrame to a CSV file without the row index
df.to_csv("aqnow_cleaned.csv", index=False)
