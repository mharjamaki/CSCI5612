import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder



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
df = pd.merge(i25, aqnow, on='Date', how='inner')
# drop 'Date' from the dataframe
df = df.drop(['Date'], axis=1)

# Count how many times each value occurs in 'CategoryName'
category_counts = df['CategoryName'].value_counts()
print(category_counts)

# Print the dataframe
print(f"Merged Dataframe:\n{df}")
# save to csv
df.to_csv('formatted_data_SVM_project.csv', index=False)

# Get the class names for plotting later
ClassNames = sorted(df['CategoryName'].unique())

# Separate the data from the label
X = df.drop(columns=['CategoryName'])
y = df['CategoryName']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=36)
test_data = pd.concat([X_test, y_test], axis=1)
train_data = pd.concat([X_train, y_train], axis=1)
# X_train.to_csv('SVMTrainingData_project.csv', index=False)
# X_test.to_csv('SVMTestingData_project.csv', index=False)
# y_train.to_csv('SVMLabel_project.csv', index=False)
# y_test.to_csv('SVMLabel_project.csv', index=False)
test_data.to_csv('test_data_SVM_project.csv', index=False)
train_data.to_csv('train_data_SVM_project.csv', index=False)

# SVM with Polynomial Kernel ###########################################################################################
SVM_Model1 = SVC(C=1, kernel='poly', degree=2, gamma="auto")
SVM_Model1.fit(X_train, y_train)

print("SVM Predicted Labels:\n", SVM_Model1.predict(X_test))
print("Actual Labels:")
print(y_test)

SVM_matrix1 = confusion_matrix(y_test, SVM_Model1.predict(X_test))
print("\nThe confusion matrix for poly p = 3 SVM is:")
print(SVM_matrix1)
print("\n\n")

sns.heatmap(SVM_matrix1, annot=True, cmap='Blues', xticklabels=ClassNames,
            yticklabels=ClassNames, cbar=False)
plt.title("Confusion Matrix for SVM with Polynomial Kernel", fontsize=20)
plt.xlabel("Actual", fontsize=15)
plt.ylabel("Predicted", fontsize=15)
plt.show()

y_pred_poly = SVM_Model1.predict(X_test)
print("Classification Report for Polynomial SVM:\n")
print(classification_report(y_test, y_pred_poly, target_names=ClassNames))


# SVM with RBF Kernel ##################################################################################################
SVM_Model2 = SVC(C=10, kernel='rbf', degree=3, gamma="auto")
SVM_Model2.fit(X_train, y_train)

print("SVM Predicted Labels:\n", SVM_Model2.predict(X_test))
print("Actual Labels:")
print(y_test)

SVM_matrix2 = confusion_matrix(y_test, SVM_Model2.predict(X_test))
print("\nThe confusion matrix for rbf SVM is:")
print(SVM_matrix2)
print("\n\n")

sns.heatmap(SVM_matrix2, annot=True, cmap='Blues', xticklabels=ClassNames,
            yticklabels=ClassNames, cbar=False)
plt.title("Confusion Matrix for SVM with RBF Kernel", fontsize=20)
plt.xlabel("Actual", fontsize=15)
plt.ylabel("Predicted", fontsize=15)
plt.show()

y_pred_rbf = SVM_Model2.predict(X_test)
print("Classification Report for RBF SVM:\n")
print(classification_report(y_test, y_pred_rbf, target_names=ClassNames))


# SVM with Linear Kernel ###############################################################################################
SVM_Model3 = SVC(C=10, kernel='linear', gamma="auto")
SVM_Model3.fit(X_train, y_train)

print("SVM Predicted Labels:\n", SVM_Model3.predict(X_test))
print("Actual Labels:")
print(y_test)

SVM_matrix3 = confusion_matrix(y_test, SVM_Model3.predict(X_test))
print("\nThe confusion matrix for linear SVM is:")
print(SVM_matrix3)
print("\n\n")

sns.heatmap(SVM_matrix3, annot=True, cmap='Blues', xticklabels=ClassNames,
            yticklabels=ClassNames, cbar=False)
plt.title("Confusion Matrix for SVM with Linear Kernel", fontsize=20)
plt.xlabel("Actual", fontsize=15)
plt.ylabel("Predicted", fontsize=15)
plt.show()

y_pred_linear = SVM_Model3.predict(X_test)
print("Classification Report for Linear SVM:\n")
print(classification_report(y_test, y_pred_linear, target_names=ClassNames))


# Visualize ############################################################################################################
# Convert categorical labels to numbers for color plotting
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Re-train model on full dataset (for cleaner boundary visualization)
svm_model = SVM_Model3
svm_model.fit(X, y_encoded)

# Define the feature range
x_min, x_max = X['PM10'].min() - 1, X['PM10'].max() + 1
y_min, y_max = X['PM2.5'].min() - 1, X['PM2.5'].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))

# Predict on the meshgrid
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

# Plot actual data points
scatter = plt.scatter(X['PM10'], X['PM2.5'], c=y_encoded, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel('PM10')
plt.ylabel('PM2.5')
plt.title('SVM Decision Boundary')
handles, labels = scatter.legend_elements()
plt.legend(handles=handles, labels=list(le.classes_))
plt.show()
