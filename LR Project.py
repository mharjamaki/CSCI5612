from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
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

# Normalize 'PM2.5' and 'PM10' columns
scaler = MinMaxScaler()
df[['PM10', 'PM2.5']] = scaler.fit_transform(df[['PM10', 'PM2.5']])

# Count how many times each value occurs in 'CategoryName'
category_counts = df['CategoryName'].value_counts()
print(category_counts)

# Fit and transform the 'CategoryName' column to numeric values
label_encoder = LabelEncoder()
df['CategoryName'] = label_encoder.fit_transform(df['CategoryName'])

# Count how many times each value occurs in 'CategoryName'
category_counts = df['CategoryName'].value_counts()
print(category_counts)

# Print the dataframe
print(f"Merged Dataframe:\n{df}")
# save to csv
df.to_csv('formatted_data_LR.csv', index=False)

# Separate the data from the label
X = df.drop(columns=['CategoryName'])
y = df['CategoryName']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
test_data = pd.concat([X_test, y_test], axis=1)
train_data = pd.concat([X_train, y_train], axis=1)
test_data.to_csv('test_data_LR.csv', index=False)
train_data.to_csv('train_data_LR.csv', index=False)
print(f"Test Data:\n {test_data}")
print(f"Train Data:\n {train_data}")

# Turn the data into a numpy array
y = np.array(y)
X = np.array(X)
y_train = np.array(y_train)
X_train = np.array(X_train)
y_test = np.array(y_test)
X_test = np.array(X_test)

# Set learning rate and define the length of X as n
n = len(X)  # number of rows of entire X
LR = 1  # Learning Rate

# Set up initial weights and biases
w = np.array([[1, 1, 1, 1]])
b = 0
print(w)
print(w.shape)

# Multiply X and w (w1x1 + w2x2 + b)
print("The shape of X is\n", X_train.shape)
print("The shape of w transpose is\n", w.T.shape)
z = (X_train @ w.T) + b
print("The shape of z is\n", z.shape)

# Apply the sigmoid function to all the z value results
def sigmoid(s, deriv=False):
    if deriv == True:
        return s * (1 - s)
    return 1 / (1 + np.exp(-s))


# Create S_z by applying the sigmoid to all the values in z
S_z = sigmoid(z)  # same as y^
print("S(z) is\n", S_z)  # the output of the logistic regression
y_hat = S_z

# Loss Categorical Entropy function
# LCE = -1/n SUM ylog(y^) + (1 - y)log(1 - y^),
# y^ is the predicted value and the log is log base e. The y is the label. The "n" is the number of rows in the dataset.
# We want to minimize the LCE by updating w and b using gradient descent

# Transpose y
print("y is\n", y_train)
print("y_hat is\n", y_hat)
y_train = np.transpose([y_train])
print("Updated y is\n", y_train)
print("y_hat is\n", y_hat)

# Keep each LCE value
AllError_LCE = []

# The epochs are the number of iterations we want to go through to recalculate w and b with the goal of optimization
# (minimization of LCE)
epochs = 200

for i in range(epochs):
    z = (X_train @ w.T) + b
    # print("The z here is\n", z)
    y_hat = sigmoid(z)
    # print("The y_hat here is\n", y_hat)

    # Get the LCE....
    # Step 1
    Z1 = (y_train * np.log(y_hat)) + ((1 - y_train) * np.log(1 - y_hat))
    # print("Z1 is\n", Z1)
    # Step 2
    # Sum the values in Z1 and then divide by n
    LCE = -(np.sum(Z1)) / n
    # print("The LCE for epoch ", i, "is\n", LCE)

    # Keep each LCE value - each error
    AllError_LCE.append(LCE)

    # Derivatives so we can update w and b
    # dL/dw = dL/dy^ * dy^/dz * dz/dw --> 1/n (y^ - y)xT
    # dL/db = dL/dy^ * dy^/dz * dz/db --> 1/n (y^ - y)

    error = y_hat - y_train
    # print("The error y^ - y is\n", error)

    # Multiply the y^-y by X so that we get the shape of w
    # print(w)
    # print(w.shape)

    dL_dw = (1 / n) * np.transpose(error) @ X_train
    # print("The dL_dw is\n", dL_dw, "\n")

    # For b, we will use the average - so we will sum up all the error values and then multiply by 1/n
    b1 = (1 / n) * (error)
    dL_db = np.average(b1)
    # print("The dL_db is\n", dL_db)

    # print("The update for w is\n", dL_dw)
    # print("The update for b is\n", dL_db)

    # Use the gradient to update w and b
    w = w - (LR * dL_dw)
    b = b - (LR * dL_db)

    # print("The new w value is\n", w)

# print(len(AllError_LCE))

# Plot the change in Loss over epochs
fig1 = plt.figure()
plt.title("Loss Reduction Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Error")
ax = plt.axes()
x = np.linspace(0, epochs, epochs)  # start, stop, how many
ax.plot(x, AllError_LCE)

print("The predicted w is \n", w)
print("The predicted b is\n", b)

# Use the model from above to make predictions.
Prediction = sigmoid((X_test @ w.T) + b)

# Update prediction using threshold >=.5 --> 1, else 0
Prediction[Prediction >= .5] = 1
Prediction[Prediction < .5] = 0

print(Prediction)

# Confusion matrix
fig2 = plt.figure()
plt.title("Confusion Matrix For Predictions vs. Actual Values")
cm = confusion_matrix(y_test, Prediction)
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
