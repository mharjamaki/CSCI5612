import pandas as pd
import missingno as msno
from matplotlib import pyplot as plt
import seaborn as sns

file_path = "C:/Users/mikyl/OneDrive/Documents/CSCI/Raw Data/I25Den_202411-12.csv"

# Load data
df = pd.read_csv(file_path, header=None)
print(df.head())

# Drop first row
df = df.drop(0)
df = df.reset_index(drop=True)
print(df)

# Replace the value at index 0, column 0 with 'Date'
df.at[0, 0] = 'Date'
print(df)

# Extract units (row 1)
units = df.iloc[1]
print(units)

# Drop the row with the units and reset index
df = df.drop(1)
df = df.reset_index(drop=True)
print(df)

# Set column names
df.columns = ['Date', 'CO TRACE', 'NO', 'NO2', 'NOX', 'WS', 'WD', 'RH', 'TEMP', 'PM10', 'PM2.5']
print(df.head())

# Drop the first row which now contains headers
df = df.drop(0)
df = df.reset_index(drop=True)
print(df)

# Calculate the count of missing values per column
missing_count = df.isnull().sum()
print("Missing count:\n", missing_count)

# Calculate the percentage of missing values per column
missing_percentage = df.isnull().mean() * 100
print("Missing percentage:\n", missing_percentage)

# visualize label as a matrix
msno.matrix(df)
plt.title('Matrix Depicting Missing Values')
plt.tight_layout()
plt.subplots_adjust(top=0.9)  # Adjust the top space, you can tweak the value if necessary
# plt.show()

# Drop 'CO TRACE'
df = df.drop(['CO TRACE'], axis=1)
print("DataFrame after dropping 'CO TRACE':\n", df.head())

# Print the data types before conversion
print("Data types before conversion:\n", df.dtypes)

columns_to_convert = ['NO', 'NO2', 'NOX', 'WS', 'WD', 'RH', 'TEMP', 'PM10', 'PM2.5']
df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')

# Check the data types after conversion
print("Data types after conversion:\n", df.dtypes)

pm_columns = ['Date', 'WS', 'WD', 'RH', 'TEMP', 'PM10', 'PM2.5']
pm_df = df[pm_columns]
print(pm_df)

# Calculate the threshold for NaN values in a row
threshold = len(pm_df.columns) / 3

# Drop rows where more than a third the values are NaN
pm_df = pm_df[pm_df.isnull().sum(axis=1) <= threshold]

# Calculate the count of missing values per column
missing_count = pm_df.isnull().sum()
print("Missing count:\n", missing_count)

# Calculate the percentage of missing values per column
missing_percentage = pm_df.isnull().mean() * 100
print("Missing percentage:\n", missing_percentage)

# For each column, identify consecutive NaN values
max_consecutive_nans_per_column = pm_df.apply(
    lambda col: col.isnull().groupby(col.isnull().cumsum()).cumsum().max()
)

print(max_consecutive_nans_per_column)

# Interpolate all columns except 'Date'
for column in pm_df.columns:
    if column != 'Date':  # Skip the 'Date' column
        pm_df[column] = pm_df[column].interpolate(method='linear', axis=0)

# Calculate the count of missing values per column
missing_count = pm_df.isnull().sum()
print("Missing count:\n", missing_count)

# Calculate the percentage of missing values per column
missing_percentage = pm_df.isnull().mean() * 100
print("Missing percentage:\n", missing_percentage)
print(pm_df)

# Boxplot for each column
# Set the number of subplots per row
num_columns = len(pm_df.columns) - 1  # Exclude 'Date'
columns_per_row = 3

# Calculate the number of rows needed
num_rows = (num_columns // columns_per_row) + (num_columns % columns_per_row > 0)

# Create subplots
fig, axes = plt.subplots(nrows=num_rows, ncols=columns_per_row, figsize=(15, 6 * num_rows))

# Flatten the axes array if it's a multi-dimensional array
axes = axes.flatten()

# Custom y-axis labels for each plot
custom_y_labels = {
    'WS': 'Wind Speed (MPH)',
    'WD': 'Wind Direction (°)',
    'RH': 'Relative Humidity (%)',
    'TEMP': 'Temperature (°F)',
    'PM10': 'Concentration (µg/m³)',
    'PM2.5': 'Concentration (µg/m³)'
}

# Custom titles
custom_titles = {
    'WS': 'Wind Speed',
    'WD': 'Wind Direction',
    'RH': 'Relative Humidity',
    'TEMP': 'Temperature',
    'PM10': 'PM10 Concentration',
    'PM2.5': 'PM2.5 Concentration'
}

# Loop through each column and create a boxplot in a separate subplot
for i, column in enumerate(pm_df.columns[1:]):  # Exclude 'Date'
    sns.boxplot(data=pm_df[column], ax=axes[i])

    # Set custom title and y-axis label for each plot
    axes[i].set_title(custom_titles.get(column))
    axes[i].set_ylabel(custom_y_labels.get(column))

plt.tight_layout()
plt.subplots_adjust(hspace=0.5)  # Adjust vertical spacing between plots
# plt.show()

# Count the number of values in each column
column_counts = pm_df.count()
print(column_counts)

# Filter rows where 'PM10' exceeds 500
exceeding_500 = pm_df[pm_df['PM10'] > 500]
print(exceeding_500)

# Calculate the average of 'PM10' for values less than or equal to 500
average_pm10 = pm_df[pm_df['PM10'] <= 500]['PM10'].mean()

# Replace values in 'PM10' that exceed 500 with the calculated average
pm_df.loc[pm_df['PM10'] > 500, 'PM10'] = average_pm10
print(pm_df)

# Boxplot for each column
# Set the number of subplots per row
num_columns = len(pm_df.columns) - 1  # Exclude 'Date'
columns_per_row = 3

# Calculate the number of rows needed
num_rows = (num_columns // columns_per_row) + (num_columns % columns_per_row > 0)

# Create subplots
fig, axes = plt.subplots(nrows=num_rows, ncols=columns_per_row, figsize=(15, 6 * num_rows))

# Flatten the axes array if it's a multi-dimensional array
axes = axes.flatten()

# Custom y-axis labels for each plot
custom_y_labels = {
    'WS': 'Wind Speed (MPH)',
    'WD': 'Wind Direction (°)',
    'RH': 'Relative Humidity (%)',
    'TEMP': 'Temperature (°F)',
    'PM10': 'Concentration (µg/m³)',
    'PM2.5': 'Concentration (µg/m³)'
}

# Custom titles
custom_titles = {
    'WS': 'Wind Speed',
    'WD': 'Wind Direction',
    'RH': 'Relative Humidity',
    'TEMP': 'Temperature',
    'PM10': 'PM10 Concentration',
    'PM2.5': 'PM2.5 Concentration'
}

# Loop through each column and create a boxplot in a separate subplot
for i, column in enumerate(pm_df.columns[1:]):  # Exclude 'Date'
    sns.boxplot(data=pm_df[column], ax=axes[i])

    # Set custom title and y-axis label for each plot
    axes[i].set_title(custom_titles.get(column))
    axes[i].set_ylabel(custom_y_labels.get(column))

plt.tight_layout()
plt.subplots_adjust(hspace=0.5)  # Adjust vertical spacing between plots
plt.show()

# Save clean data to CSV
pm_df.to_csv("I25_cleaned.csv", index=False)
