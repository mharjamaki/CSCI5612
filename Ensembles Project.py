from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load Traffic Data
traffic = pd.read_csv("C:/Users/mikyl/OneDrive - UCB-O365/Class Documents/CSCI/Cleaned Data/traffic_cleaned.csv")
traffic = traffic[['WEEKEND', 'COUNTDATE']]
traffic.rename(columns={'COUNTDATE': 'Date'}, inplace=True)
traffic['Date'] = pd.to_datetime(traffic['Date'])

# Load I25 Data
i25 = pd.read_csv("C:/Users/mikyl/OneDrive - UCB-O365/Class Documents/CSCI/Cleaned Data/I25_cleaned.csv")
i25 = i25[['Date', 'PM10', 'PM2.5']]
i25['Date'] = pd.to_datetime(i25['Date'])
i25['Date'] = i25['Date'].dt.date
i25 = i25.groupby('Date').agg({'PM10': 'mean', 'PM2.5': 'mean'}).reset_index()
i25['Date'] = pd.to_datetime(i25['Date'])

# Load AQNow Data
aqnow = pd.read_csv("C:/Users/mikyl/OneDrive - UCB-O365/Class Documents/CSCI/Cleaned Data/aqnow_cleaned.csv")
aqnow.rename(columns={'DateObserved': 'Date'}, inplace=True)
aqnow['Date'] = pd.to_datetime(aqnow['Date'])
idx = aqnow.groupby('Date')['AQI'].idxmax()
aqnow = aqnow.loc[idx]
aqnow = aqnow[['Date', 'CategoryName']]

# Merge all data
df = pd.merge(traffic, i25, on='Date', how='inner')
df = pd.merge(df, aqnow, on='Date', how='inner')
df.drop(['Date'], axis=1, inplace=True)
df.to_csv('Ensemble_Project_Data.csv', index=False)

# # Encode label
# label_encoder = LabelEncoder()
# df['CategoryName'] = label_encoder.fit_transform(df['CategoryName'])
# category_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
# print("Category to encoded label mapping:", category_mapping)

# Prepare data for training
X = df.drop(columns=['CategoryName'])
y = df['CategoryName']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=35)

test_data = pd.concat([X_test, y_test], axis=1)
train_data = pd.concat([X_train, y_train], axis=1)
test_data.to_csv('test_data_ensemble_project.csv', index=False)
train_data.to_csv('train_data_ensemble_project.csv', index=False)

# Perform Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
class_names = [str(cls) for cls in rf_classifier.classes_]

sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=False)
plt.title("Confusion Matrix for Random Forest", fontsize=20)
plt.xlabel("Predicted", fontsize=15)
plt.ylabel("Actual", fontsize=15)
plt.show()

# Classification report
print("Classification Report for Random Forest:\n")
print(classification_report(y_test, y_pred, target_names=class_names))
