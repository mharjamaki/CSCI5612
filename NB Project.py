import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder


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

# LOW-COST SENSOR DATA
mini = pd.read_csv("C:/Users/mikyl/OneDrive - UCB-O365/Class Documents/CSCI/Cleaned Data/minipod_cleaned.csv")
# drop all columns except 'Date', 'PM10', and 'PM2.5'
mini = mini[['DateTime', 'PM 10.0', 'PM2.5']]
# rename 'DateTime' to 'Date'
mini.rename(columns={'DateTime': 'Date'}, inplace=True)
# convert 'Date' column to datetime
mini['Date'] = pd.to_datetime(mini['Date'])
# extract only the date from the datetime
mini['Date'] = mini['Date'].dt.date
# group by the extracted 'Date' and calculate the daily average for 'PM10' and 'PM2.5'
mini = mini.groupby('Date').agg({'PM 10.0': 'mean', 'PM2.5': 'mean'}).reset_index()
# convert 'Date' back to datetime
mini['Date'] = pd.to_datetime(mini['Date'])
# print dataframe
# print(mini)

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
df = pd.merge(traffic, mini, on='Date', how='inner')
df = pd.merge(df, aqnow, on='Date', how='inner')
df = df.drop(['Date'], axis=1)
df.to_csv('ALL_NB_formatted_data.csv', index=False)

#######################################################################################################################
# Multinomial Naive Bayes

# Round to the nearest integer for MN naive bayes
mini_MNB = mini
mini_MNB['PM 10.0'] = mini_MNB['PM 10.0'].round()
mini_MNB['PM2.5'] = mini_MNB['PM2.5'].round()
# print data frame
# print(mini_MNB)

# Merge all the data on 'Date'
df_MNB = pd.merge(traffic, mini_MNB, on='Date', how='inner')
df_MNB = pd.merge(df_MNB, aqnow, on='Date', how='inner')
df_MNB = df_MNB.drop(['Date'], axis=1)
df_MNB.to_csv('NB_formatted_data.csv', index=False)
# print(df_MNB)

# Count how many times each value occurs in 'CategoryName'
category_counts_MNB = df_MNB['CategoryName'].value_counts()
print(category_counts_MNB)

# Fit and transform the 'CategoryName' column to numeric values
label_encoder = LabelEncoder()
df_MNB['CategoryName'] = label_encoder.fit_transform(df_MNB['CategoryName'])
category_counts = df_MNB['CategoryName'].value_counts()
print(category_counts)
# print(df_MNB)
df_MNB.to_csv('formatted_data_LCS_MNB.csv', index=False)

# Separate the data from the label
X_MNB = df_MNB.drop(columns=['CategoryName'])
y_MNB = df_MNB['CategoryName']

# Split the data into training and testing sets (80% training, 20% testing)
X_train_MNB, X_test_MNB, y_train_MNB, y_test_MNB = train_test_split(X_MNB, y_MNB, test_size=0.15)
test_data_MNB = pd.concat([X_test_MNB, y_test_MNB], axis=1)
train_data_MNB = pd.concat([X_train_MNB, y_train_MNB], axis=1)
test_data_MNB.to_csv('test_data_MNB.csv', index=False)
train_data_MNB.to_csv('train_data_MNB.csv', index=False)
print(f"Test Data:\n {test_data_MNB}")
print(f"Train Data:\n {train_data_MNB}")

# Turn the data into a numpy array
y_MNB = np.array(y_MNB)
X_MNB = np.array(X_MNB)
y_train_MNB = np.array(y_train_MNB)
X_train_MNB = np.array(X_train_MNB)
y_test_MNB = np.array(y_test_MNB)
X_test_MNB = np.array(X_test_MNB)

# Instantiate the Multinomial Naive Bayes model
MyMN = MultinomialNB()

# Train the model
My_MN_Model = MyMN.fit(X_train_MNB, y_train_MNB)
print("Trained Multinomial Naive Bayes Model:", My_MN_Model)
print("Classes in the model:", My_MN_Model.classes_)

# Predict the Testing Data using the model
Predictions_MNB = My_MN_Model.predict(X_test_MNB)
print("Predictions on Testing Data:", Predictions_MNB)

# Print the actual probabilities
print("The Multinomial NB Model Prediction Probabilities are:")
print(My_MN_Model.predict_proba(X_test_MNB).round(3))

# Confusion Matrix
fig1 = plt.figure()
plt.title("Confusion Matrix For Predictions vs. Actual Values")
cm = confusion_matrix(y_test_MNB, Predictions_MNB)
print(cm)
ax = plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='Blues')
# annot=True to annotate cells, ftm='g' to disable scientific notation
ax.set_xlabel("Predicted labels")
ax.set_ylabel("True labels")
ax.set_title("Confusion Matrix for Multinomial Naive Bayes")
ax.xaxis.set_ticklabels(["Moderate", "Good"])
ax.yaxis.set_ticklabels(["Moderate", "Good"])
plt.show()


#######################################################################################################################
# Gaussian Naive Bayes
df_GNB = pd.merge(mini, aqnow, on='Date', how='inner')
# drop 'Date' from the dataframe
df_GNB = df_GNB.drop(['Date'], axis=1)

# Count how many times each value occurs in 'CategoryName'
category_counts_GNB = df_GNB['CategoryName'].value_counts()
print(category_counts_GNB)

# Fit and transform the 'CategoryName' column to numeric values
label_encoder = LabelEncoder()
df_GNB['CategoryName'] = label_encoder.fit_transform(df_GNB['CategoryName'])
category_counts = df_GNB['CategoryName'].value_counts()
print(category_counts)
print(df_GNB)
df_GNB.to_csv('formatted_data_GNB.csv', index=False)

# Separate the data from the label
X_GNB = df_GNB.drop(columns=['CategoryName'])
y_GNB = df_GNB['CategoryName']

# Split the data into training and testing sets (80% training, 20% testing)
X_train_GNB, X_test_GNB, y_train_GNB, y_test_GNB = train_test_split(X_GNB, y_GNB, test_size=0.15)
test_data_GNB = pd.concat([X_test_GNB, y_test_GNB], axis=1)
train_data_GNB = pd.concat([X_train_GNB, y_train_GNB], axis=1)
test_data_GNB.to_csv('test_data_GNB.csv', index=False)
train_data_GNB.to_csv('train_data_GNB.csv', index=False)
print(f"Test Data:\n {test_data_GNB}")
print(f"Training Data:\n {train_data_GNB}")

# Turn the data into a numpy array
y_GNB = np.array(y_GNB)
X_GNB = np.array(X_GNB)
y_train_GNB = np.array(y_train_GNB)
X_train_GNB = np.array(X_train_GNB)
y_test_GNB = np.array(y_test_GNB)
X_test_GNB = np.array(X_test_GNB)

# Instantiate
MyGNB = GaussianNB()

# Training the model
My_GNB_Model = MyGNB.fit(X_train_GNB, y_train_GNB)
print(My_GNB_Model)

# Predict the Testing Data using the model
Predictions_G=My_GNB_Model.predict(X_test_GNB)
print(Predictions_G)

# Print the probabilities
print("The Gaussian NB Model Prediction Probabilities are:")
print(My_GNB_Model.predict_proba(X_test_GNB).round(3))

# Gaussian Naive Bayes Confusion Matrix
fig2 = plt.figure()
plt.title("Confusion Matrix For Predictions vs. Actual Values")
cm = confusion_matrix(y_test_GNB, Predictions_G)
print(cm)
ax = plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='Blues')
# annot=True to annotate cells, ftm='g' to disable scientific notation
ax.set_xlabel("Predicted labels")
ax.set_ylabel("True labels")
ax.set_title("Confusion Matrix for Gaussian Naive Bayes")
ax.xaxis.set_ticklabels(["Moderate", "Good"])
ax.yaxis.set_ticklabels(["Moderate", "Good"])
plt.show()


#######################################################################################################################
# Bernoulli Naive Bayes

# Merge all the data on 'Date'
df_BNB = pd.merge(traffic, aqnow, on='Date', how='inner')
# drop 'Date' from the dataframe
df_BNB = df_BNB.drop(['Date'], axis=1)

# Group 'Unhealthy for Sensitive Groups' and 'Unhealthy' together
df_BNB['CategoryName'] = df_BNB['CategoryName'].replace('Unhealthy for Sensitive Groups', 'Unhealthy')

# Count how many times each value occurs in 'CategoryName'
category_counts = df_BNB['CategoryName'].value_counts()
print(category_counts)

# Resample the data so each category has 39 rows
df_BNB = df_BNB.groupby('CategoryName').apply(lambda x: x.sample(n=39, replace=True))

# Reset the index after resampling
df_BNB.reset_index(drop=True, inplace=True)

# Count how many times each value occurs in 'CategoryName' in the resampled data
category_counts = df_BNB['CategoryName'].value_counts()
print(category_counts)

# Initialize the label encoder
label_encoder = LabelEncoder()
# Encode 'CategoryName' into numerical values
df_BNB['CategoryName'] = label_encoder.fit_transform(df_BNB['CategoryName'])
# Print the mapping of labels to encoded values
print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
category_counts = df_BNB['CategoryName'].value_counts()
print(category_counts)
df_BNB.to_csv('formatted_data_BNB.csv', index=False)

# Separate the data from the label
X_BNB = df_BNB.drop(columns=['CategoryName'])
y_BNB = df_BNB['CategoryName']

# Split the data into training and testing sets (80% training, 20% testing)
X_train_BNB, X_test_BNB, y_train_BNB, y_test_BNB = train_test_split(X_BNB, y_BNB, test_size=0.15)
test_data_BNB = pd.concat([X_test_BNB, y_test_BNB], axis=1)
train_data_BNB = pd.concat([X_train_BNB, y_train_BNB], axis=1)
test_data_BNB.to_csv('test_data_BNB.csv', index=False)
train_data_BNB.to_csv('train_data_BNB.csv', index=False)
print(f"Testing Data:\n {test_data_BNB}")
print(f"Training Data:\n {train_data_BNB}")

# Turn the data into a numpy array
y_BNB = np.array(y_BNB)
X_BNB = np.array(X_BNB)
y_train_BNB = np.array(y_train_BNB)
X_train_BNB = np.array(X_train_BNB)
y_test_BNB = np.array(y_test_BNB)
X_test_BNB = np.array(X_test_BNB)

# Instantiate first
MyBNB = BernoulliNB()

# Training the model
My_BNB_Model = MyBNB.fit(X_train_BNB, y_train_BNB)
print(My_BNB_Model)
print(My_BNB_Model.classes_)

# Predict the Testing Data using the model
Predictions_B = My_BNB_Model.predict(X_test_BNB)
print(Predictions_B)

# Print the actual probabilities
print("The Bernoulli NB Model Prediction Probabilities are:")
print(My_BNB_Model.predict_proba(X_test_BNB).round(3))

# Confusion Matrix
fig3 = plt.figure()
plt.title("Confusion Matrix For Predictions vs. Actual Values")
cm = confusion_matrix(y_test_BNB, Predictions_B)
print(cm)
ax = plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='Blues')
# annot=True to annotate cells, ftm='g' to disable scientific notation
ax.set_xlabel("Predicted labels")
ax.set_ylabel("True labels")
ax.set_title("Confusion Matrix for Bernoulli Naive Bayes")
ax.xaxis.set_ticklabels(["Unhealthy", "Moderate", "Good"])
ax.yaxis.set_ticklabels(["Unhealthy", "Moderate", "Good"])
plt.show()
