import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import seaborn as sns


# TRAFFIC DATA
# read in traffic data
traffic = pd.read_csv("C:/Users/mikyl/OneDrive - UCB-O365/Class Documents/CSCI/Cleaned Data/traffic_cleaned.csv")
# drop all columns except 'WEEKEND' and 'COUNTDATE'
traffic = traffic[['WEEKEND', 'COUNTDATE']]
# rename 'COUNTDATE' to 'Date'
traffic.rename(columns={'COUNTDATE': 'Date'}, inplace=True)
# convert 'Date' column to datetime
traffic['Date'] = pd.to_datetime(traffic['Date'])
# one-hot encode 'WEEKEND'
traffic = pd.get_dummies(traffic, columns=['WEEKEND'], dtype=int)
# print formatted dataframe
# print(traffic)

# I25 DATA
i25 = pd.read_csv("C:/Users/mikyl/OneDrive - UCB-O365/Class Documents/CSCI/Cleaned Data/I25_cleaned.csv")
# drop all columns except 'Date', 'PM10', and 'PM2.5'
i25 = i25[['Date', 'PM10', 'PM2.5']]
# convert 'Date' column to datetime
i25['Date'] = pd.to_datetime(i25['Date'])
# extract only the date from the datetime
i25['Date'] = i25['Date'].dt.date
# group by the extracted 'Date' and calculate the daily average for 'PM10' and 'PM2.5'
i25 = i25.groupby('Date').agg({'PM10': 'mean', 'PM2.5': 'mean'}).reset_index()
# convert 'Date' back to datetime
i25['Date'] = pd.to_datetime(i25['Date'])
# round to the nearest integer for MN naive bayes
i25['PM10'] = i25['PM10'].round()
i25['PM2.5'] = i25['PM2.5'].round()
# print the formatted dataframe
# print(i25)

# AQNOW DATA
# read in aqnow data
aqnow = pd.read_csv("C:/Users/mikyl/OneDrive - UCB-O365/Class Documents/CSCI/Cleaned Data/aqnow_cleaned.csv")
# rename 'DateObserved' to 'Date'
aqnow.rename(columns={'DateObserved': 'Date'}, inplace=True)
# convert 'Date' column to datetime
aqnow['Date'] = pd.to_datetime(aqnow['Date'])
# group by 'Date' and get the row with the highest 'AQI' for each date
idx = aqnow.groupby('Date')['AQI'].idxmax()
# get the rows with the highest AQI for each date
aqnow = aqnow.loc[idx]
# drop all columns except 'Date' and 'CategoryName'
aqnow = aqnow[['Date', 'CategoryName']]
# print the formatted dataframe
# print(aqnow)

# Merge all the data on 'Date'
df = pd.merge(traffic, i25, on='Date', how='inner')
df = pd.merge(df, aqnow, on='Date', how='inner')
# drop 'Date' from the dataframe
df = df.drop(['Date'], axis=1)
# print the dataframe
# print(df)

# Count how many times each value occurs in 'CategoryName'
category_counts = df['CategoryName'].value_counts()
print(category_counts)

# Fit and transform the 'CategoryName' column to numeric values
label_encoder = LabelEncoder()
df['CategoryName'] = label_encoder.fit_transform(df['CategoryName'])
category_counts = df['CategoryName'].value_counts()
print(category_counts)
print(f"Merged Dataframe: {df}")
df.to_csv('formatted_data_HCS_MNB.csv', index=False)

# Separate the data from the label
X = df.drop(columns=['CategoryName'])
y = df['CategoryName']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
test_data = pd.concat([X_test, y_test], axis=1)
train_data = pd.concat([X_train, y_train], axis=1)
test_data.to_csv('test_data_HCS_MNB.csv', index=False)
train_data.to_csv('train_data_HCS_MNB.csv', index=False)
print(f"Training Data:\n {train_data}")
print(f"Test Data:\n {test_data}")

# Turn the data into a numpy array
y = np.array(y)
X = np.array(X)
y_train = np.array(y_train)
X_train = np.array(X_train)
y_test = np.array(y_test)
X_test = np.array(X_test)

# Instantiate the Multinomial Naive Bayes model
MyMN = MultinomialNB()

# Train the model
My_MN_Model = MyMN.fit(X_train, y_train)
print("Trained Multinomial Naive Bayes Model:", My_MN_Model)
print("Classes in the model:", My_MN_Model.classes_)

# Predict the Testing Data using the model
Predictions_MN = My_MN_Model.predict(X_test)
print("Predictions on Testing Data:", Predictions_MN)

# Print the actual probabilities
print("The Multinomial NB Model Prediction Probabilities are:")
print(My_MN_Model.predict_proba(X_test).round(3))

# Confusion Matrix
fig2 = plt.figure()
plt.title("Confusion Matrix For Predictions vs. Actual Values")
cm = confusion_matrix(y_test, Predictions_MN)
print(cm)
ax = plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='Blues')
# annot=True to annotate cells, ftm='g' to disable scientific notation
ax.set_xlabel("Predicted labels")
ax.set_ylabel("True labels")
ax.set_title("Confusion Matrix")
ax.xaxis.set_ticklabels(["Moderate", "Good"])
ax.yaxis.set_ticklabels(["Moderate", "Good"])
plt.show()
