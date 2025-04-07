from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
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

# Initialize the LabelEncoder
label_encoder = LabelEncoder()
# Apply label encoding to 'CategoryName'
df['CategoryName'] = label_encoder.fit_transform(df['CategoryName'])
# Map the encoded labels back to the original categories
category_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
# Display the mapping of categories to encoded labels
print("Category to encoded label mapping:", category_mapping)
# Count how many times each value occurs in 'CategoryName'
category_counts = df['CategoryName'].value_counts()
print(f"Category Counts:\n {category_counts}")

print(f"Merged Dataframe: \n {df}")
print(df.dtypes)
df.to_csv('DT_formatted_data.csv', index=False)

# Separate the data from the label
X = df.drop(columns=['CategoryName'])
y = df['CategoryName']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
test_data = pd.concat([X_test, y_test], axis=1)
train_data = pd.concat([X_train, y_train], axis=1)
test_data.to_csv('test_data_DT.csv', index=False)
train_data.to_csv('train_data_DT.csv', index=False)
print(f"Test Data:\n {test_data}")
print(f"Train Data:\n {train_data}")

# Default Options #####################################################################################################
# Instantiate the decision tree using the defaults.
MyDT_Classifier = DecisionTreeClassifier()

# Use fit to create the decision tree (DT) model
MyDT_Classifier = MyDT_Classifier.fit(X_train, y_train)

# Get all the feature/variable names for visualizations
FeatureNames = X_train.columns.values  # get all the feature/variable names
print(FeatureNames)
ClassNames = MyDT_Classifier.classes_  # Get the class names
print(ClassNames)
ClassNames = [str(cls) for cls in ClassNames]

# Tree Plot Option
MyPlot = tree.plot_tree(MyDT_Classifier,
                        feature_names=FeatureNames,
                        class_names=ClassNames,
                        filled=True)
plt.savefig("Tree.jpg", dpi=300)
plt.close()  # prevents python from plotting graphs on top of each other

# Use the Tree to make predictions
Prediction = MyDT_Classifier.predict(X_test)
print(Prediction)

Actual_Labels = y_test
Predicted_Labels = Prediction

# Confusion Matrix
My_Conf_Mat = confusion_matrix(Actual_Labels, Predicted_Labels)
print(My_Conf_Mat)

sns.heatmap(My_Conf_Mat, annot=True, cmap='Blues', xticklabels=ClassNames,
            yticklabels=ClassNames, cbar=False)
plt.title("Confusion Matrix for Decision Tree", fontsize=20)
plt.xlabel("Actual", fontsize=15)
plt.ylabel("Predicted", fontsize=15)

plt.savefig("CM.jpg")
plt.close()


# Entropy + Random Splitter Option ###################################################################################
# Instantiate the decision tree using the defaults.
MyDT_Classifier2 = DecisionTreeClassifier(criterion='entropy', splitter='random')

# Use fit to create the decision tree (DT) model
MyDT_Classifier2 = MyDT_Classifier2.fit(X_train, y_train)

# Get all the feature/variable names for visualizations
FeatureNames2 = X_train.columns.values  # get all the feature/variable names
print(FeatureNames2)
ClassNames2 = MyDT_Classifier2.classes_  # Get the class names
print(ClassNames2)
ClassNames2 = [str(cls) for cls in ClassNames2]

# Tree Plot Option
MyPlot2 = tree.plot_tree(MyDT_Classifier2,
                        feature_names=FeatureNames2,
                        class_names=ClassNames2,
                        filled=True)
plt.savefig("Tree2.jpg", dpi=300)
plt.close()  # prevents python from plotting graphs on top of each other

# Use the Tree to make predictions
Prediction2 = MyDT_Classifier2.predict(X_test)
print(Prediction2)

Actual_Labels2 = y_test
Predicted_Labels2 = Prediction2

# Confusion Matrix
My_Conf_Mat2 = confusion_matrix(Actual_Labels2, Predicted_Labels2)
print(My_Conf_Mat2)

sns.heatmap(My_Conf_Mat2, annot=True, cmap='Blues', xticklabels=ClassNames2,
            yticklabels=ClassNames2, cbar=False)
plt.title("Confusion Matrix for Decision Tree", fontsize=20)
plt.xlabel("Actual", fontsize=15)
plt.ylabel("Predicted", fontsize=15)

plt.savefig("CM2.jpg")
plt.close()


# GINI + Random Splitter Option ########################################################################################
# Instantiate the decision tree using the defaults.
MyDT_Classifier3 = DecisionTreeClassifier(splitter='random', max_depth=4)

# Use fit to create the decision tree (DT) model
MyDT_Classifier3 = MyDT_Classifier3.fit(X_train, y_train)

# Get all the feature/variable names for visualizations
FeatureNames3 = X_train.columns.values  # get all the feature/variable names
print(FeatureNames3)
ClassNames3 = MyDT_Classifier3.classes_  # Get the class names
print(ClassNames3)
ClassNames3 = [str(cls) for cls in ClassNames3]

# Tree Plot Option
MyPlot3 = tree.plot_tree(MyDT_Classifier3,
                        feature_names=FeatureNames3,
                        class_names=ClassNames3,
                        filled=True)
plt.savefig("Tree3.jpg", dpi=300)
plt.close()  # prevents python from plotting graphs on top of each other

# Use the Tree to make predictions
Prediction3 = MyDT_Classifier3.predict(X_test)
print(Prediction3)

Actual_Labels3 = y_test
Predicted_Labels3 = Prediction3

# Confusion Matrix
My_Conf_Mat3 = confusion_matrix(Actual_Labels3, Predicted_Labels3)
print(My_Conf_Mat3)

sns.heatmap(My_Conf_Mat3, annot=True, cmap='Blues', xticklabels=ClassNames3,
            yticklabels=ClassNames3, cbar=False)
plt.title("Confusion Matrix for Decision Tree", fontsize=20)
plt.xlabel("Actual", fontsize=15)
plt.ylabel("Predicted", fontsize=15)

plt.savefig("CM3.jpg")
plt.close()
