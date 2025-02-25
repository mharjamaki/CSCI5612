import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define file paths for the datasets
file_path_1 = "C:/Users/mikyl/OneDrive/Documents/CSCI/Raw Data/MiniM10.csv"
file_path_2 = "C:/Users/mikyl/OneDrive/Documents/CSCI/Raw Data/Headers.csv"

# Read CSV files
df1 = pd.read_csv(file_path_1, header=None)
df2 = pd.read_csv(file_path_2, header=None)

# Concatenate the two DataFrames, stacking df2 on top of df1
df = pd.concat([df2, df1], axis=0, ignore_index=True)

# Set pandas display option to show all columns
pd.set_option('display.max_columns', None)
print(df)

# Extract units from the second row and convert it into a list
units = df.iloc[1].to_list()
print(units)

# Drop the first two rows
df = df.drop([1, 2])

# Reset the DataFrame index after dropping rows
df = df.reset_index(drop=True)

# Extract the first row as column headers and convert it to a list
headers = df.iloc[0].to_list()

# Drop the first row, which is now being used as column names
df = df.drop(0)

# Set the new column names in the DataFrame
df.columns = headers

# Calculate the reject percentage based on the 'Reject Glitch' column
total_rows = len(df['Reject Glitch'])
reject_count = (df['Reject Glitch'] > 5).sum()  # Count how many values exceed 10
reject_percentage = reject_count / total_rows  # Calculate percentage of rows exceeding 10
print(reject_percentage)

# Remove rows where 'Reject Glitch' exceeds 10 and reset index
df = df[df['Reject Glitch'] <= 5].reset_index(drop=True)

# Recalculate reject percentage based on the 'Reject Long' column
total_rows = len(df['Reject Long'])
reject_count = (df['Reject Long'] > 5).sum()  # Count how many values exceed 10
reject_percentage = reject_count / total_rows  # Calculate percentage of rows exceeding 10
print(reject_percentage)

# Remove rows where 'Reject Long' exceeds 10 and reset index
df = df[df['Reject Long'] <= 5].reset_index(drop=True)

# Convert 'DateTime' column to pandas datetime format and filter rows after a specific start time
start_time = pd.to_datetime("2024-11-18 00:00:00")
df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%dT%H:%M:%S', errors='coerce')
df = df[df['DateTime'] >= start_time].reset_index(drop=True)

# Print the data types of each column
print(df.dtypes)

# Drop PSD columns
psd_columns = ['Bin 0 ', 'Bin 1', 'Bin 2', 'Bin 3', 'Bin 4', 'Bin 5 ', 'Bin 6', 'Bin 7', 'Bin 8', 'Bin 9', 'Bin 10',
               'Bin 11', 'Bin 12', 'Bin 13', 'Bin 14', 'Bin 15', 'SampleFlowRate', 'Sample Period']
df = df.drop(psd_columns, axis=1)

# Drop other unnecessary columns
drop_columns = ['Signal Temp', 'Signal RH', 'MToF Bin1', 'MToF Bin3', 'MToF Bin5', 'MToF Bin7', 'Reject Glitch',
                'Reject Long', 'Checksum', 'Verify', 'Temperature', 'Relative Humidity']
df = df.drop(drop_columns, axis=1)

# Convert specific columns to numeric values, coercing errors
columns_to_convert = ['PM 1.0', 'PM2.5', 'PM 10.0']
df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')
print(df.dtypes)

# Count missing values for each column
missing_count = df.isnull().sum()
print(missing_count)

# Exclude the 'DateTime' column when creating boxplots
columns_to_plot = [col for col in df.columns if col != 'DateTime']

# Define a custom list of units for each column in `columns_to_plot`
custom_units = ['µg/m³', 'µg/m³', 'µg/m³']

# Create a figure with subplots (one subplot for each column to be plotted)
fig, axes = plt.subplots(nrows=1, ncols=len(columns_to_plot), figsize=(15, 5))

# If there's only one column to plot, make sure axes is iterable
if len(columns_to_plot) == 1:
    axes = [axes]

# Loop through each column and create a boxplot for it
for i, column in enumerate(columns_to_plot):
    sns.boxplot(data=df[column], ax=axes[i])
    axes[i].set_title(f'{column}')
    axes[i].set_ylabel(f'[{custom_units[i]}]')

plt.tight_layout()
plt.show()

# Filter rows where 'PM 10.0' values are greater than 2000
high_pm10_rows = df[df['PM 10.0'] > 2000]
print(high_pm10_rows)

# Specify the indices of the rows you want to delete
indices_to_delete = [37722, 124982, 181933]

# Drop the rows based on the indices
df = df.drop(indices_to_delete, axis=0).reset_index(drop=True)

# Filter rows where 'PM2.5' values are greater than 100
high_pm25_rows = df[df['PM2.5'] > 100]
print(high_pm25_rows)

# Specify the indices of the rows you want to delete
indices_to_delete = [97314, 125392, 125408, 179292]

# Drop the rows based on the indices
df = df.drop(indices_to_delete, axis=0).reset_index(drop=True)

# Filter rows where 'PM 1.0' values are greater than 15
high_pm1_rows = df[df['PM 1.0'] > 15]
print(high_pm1_rows)

# Exclude the 'DateTime' column when creating boxplots
columns_to_plot = [col for col in df.columns if col != 'DateTime']

# Define a custom list of units for each column in `columns_to_plot`
custom_units = ['µg/m³', 'µg/m³', 'µg/m³']

# Create a figure with subplots (one subplot for each column to be plotted)
fig, axes = plt.subplots(nrows=1, ncols=len(columns_to_plot), figsize=(15, 5))

# If there's only one column to plot, make sure axes is iterable
if len(columns_to_plot) == 1:
    axes = [axes]

# Loop through each column and create a boxplot for it
for i, column in enumerate(columns_to_plot):
    sns.boxplot(data=df[column], ax=axes[i])
    axes[i].set_title(f'{column}')
    axes[i].set_ylabel(f'[{custom_units[i]}]')

plt.tight_layout()
plt.show()

# Plot all three PM values on the same time series plot
plt.figure(figsize=(12, 6))
# plt.plot(df['DateTime'], df['PM 1.0'], label='PM 1.0', marker='o', markersize=4, linestyle=' ', color='blue', alpha=0.7)
# plt.plot(df['DateTime'], df['PM2.5'], label='PM 2.5', marker='o', markersize=4, linestyle=' ', color='orange', alpha=0.7)
plt.plot(df['DateTime'], df['PM 10.0'], label='PM 10.0', marker='o', markersize=4, linestyle=' ', color='green', alpha=0.7)
plt.xlabel('DateTime')
plt.ylabel('Concentration (µg/m³)')
plt.ylim(0)
plt.title('PM10 Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Save the cleaned DataFrame to a new CSV file
df.to_csv("minipod_cleaned.csv", index=False)
