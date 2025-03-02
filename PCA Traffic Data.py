import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import colors


# Load data ############################################################################################################
# Define path to file
path = "C:/Users/mikyl/OneDrive/Documents/CSCI/Cleaned Data/traffic_cleaned.csv"

# Load traffic data into a dataframe
df = pd.read_csv(path)
Original_df = df.copy()


# Format data ##########################################################################################################
# Drop 'COUNTDATE'
df.drop(columns=['COUNTDATE'], inplace=True)
df.reset_index(drop=True)
print(df)


# Remove and save the label ############################################################################################
# Save the label
df_Label = df['DAY_OF_WEEK']
print(df_Label)
print(type(df_Label))  # Check the datatype

# Remove the label from the original dataframe
df = df.drop(['DAY_OF_WEEK'], axis=1)
print(df)

# Save the label
df_Label2 = df['WEEKEND']
print(df_Label2)
print(type(df_Label2))  # Check the datatype

# Remove the label from the original dataframe
df = df.drop(['WEEKEND'], axis=1)
print(df)

# Save formatted data as csv
df.to_csv('traffic_data_pca.csv', index=False)


# Standardize your dataset #############################################################################################
scaler = StandardScaler()  # Instantiate
df = scaler.fit_transform(df)  # Scale data
print(f"Standardized Dataframe: {df}")
print(type(df))
print(df.shape)


# Perform 2D PCA #######################################################################################################
# Instantiate PCA and choose how many components
MyPCA_2D = PCA(n_components=2)

# Project the original data into the PCA space
Result_2D = MyPCA_2D.fit_transform(df)

# Print results
print(Result_2D)
print("The eigenvalues:", MyPCA_2D.explained_variance_)
print("The relative eigenvalues are:", MyPCA_2D.explained_variance_ratio_)

# Calculate the cumulative variance
cumulative_variance_2D = np.cumsum(MyPCA_2D.explained_variance_ratio_)
print("The cumulative variance is:", cumulative_variance_2D[1])


# Visualize 2D PCA #####################################################################################################
# # Visualize the cumulative variance for 2D PCA
# plt.figure(figsize=(8, 6))
# plt.plot(range(1, len(cumulative_variance_2D) + 1), cumulative_variance_2D, marker='o', linestyle='--', color='b')
# plt.title('Cumulative Explained Variance - 2D PCA')
# plt.xlabel('Number of Principal Components')
# plt.ylabel('Cumulative Explained Variance')

# Visualize 2D PCA Scatter Plot with Day of Week Labels
fig1 = plt.figure(figsize=(8, 6))
plt.scatter(Result_2D[:, 0], Result_2D[:, 1], c=df_Label, cmap="Set1", edgecolor='k', s=50)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D PCA with Day of Week Label Applied')
plt.colorbar(label='Day of Week')

# Visualize 2D PCA Scatter Plot with Weekend/Weekday Labels
fig3 = plt.figure(figsize=(8, 6))
unique_labels2 = np.unique(df_Label2)
cmap = plt.get_cmap("Set1", len(unique_labels2))
norm = colors.BoundaryNorm(np.arange(len(unique_labels2) + 1), cmap.N)
sc = plt.scatter(Result_2D[:, 0], Result_2D[:, 1], c=df_Label2, cmap=cmap, norm=norm, edgecolor='k', s=50)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D PCA with Weekday/Weekend Label Applied')
plt.colorbar(sc, label='Weekend', ticks=np.arange(len(unique_labels2)))

plt.show()


# Perform 3D PCA #######################################################################################################
# Instantiate PCA and choose how many components
MyPCA_3D = PCA(n_components=3)

# Project the original data into the PCA space
Result_3D = MyPCA_3D.fit_transform(df)
print(Result_3D)

# Print results
print("The eigenvalues:", MyPCA_3D.explained_variance_)
print("The relative eigenvalues are:", MyPCA_3D.explained_variance_ratio_)

# Calculate the cumulative variance for 3D
cumulative_variance_3D = np.cumsum(MyPCA_3D.explained_variance_ratio_)
print("The cumulative variance is:", cumulative_variance_3D[2])


# Visualize 3D PCA #####################################################################################################
# # Visualize the cumulative variance for 3D PCA
# plt.figure(figsize=(8, 6))
# plt.plot(range(1, len(cumulative_variance_3D) + 1), cumulative_variance_3D, marker='o', linestyle='--', color='r')
# plt.title('Cumulative Explained Variance - 3D PCA')
# plt.xlabel('Number of Principal Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.grid(True)
# plt.show()

# Visualize 3D PCA Scatter Plot (with df_label1)
fig2 = plt.figure()
ax2 = fig2.add_subplot(projection='3d')
x = Result_3D[:, 0]
y = Result_3D[:, 1]
z = Result_3D[:, 2]
sc3d = ax2.scatter(x, y, z, cmap="Set1", edgecolor='k', s=200, c=df_Label)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('3D PCA with Day of Week Label')
plt.colorbar(sc3d, label='Day of Week')

# Visualize 3D PCA Scatter Plot (with df_Label2)
fig4 = plt.figure()
ax4 = fig4.add_subplot(projection='3d')
x = Result_3D[:, 0]
y = Result_3D[:, 1]
z = Result_3D[:, 2]
unique_labels2 = np.unique(df_Label2)
cmap = plt.get_cmap("Set1", len(unique_labels2))
norm = colors.BoundaryNorm(np.arange(len(unique_labels2) + 1), cmap.N)
sc3d2 = ax4.scatter(x, y, z, cmap=cmap, edgecolor='k', s=200, c=df_Label2, norm=norm)
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')
ax4.set_title('3D PCA with Weekday/Weekend Label')
plt.colorbar(sc3d2, label='Weekend', ticks=np.arange(len(unique_labels2)))


# How many dimensions are needed to retain 95% of the data? ############################################################
# Perform PCA with all components
MyPCA_full = PCA()
MyPCA_full.fit(df)

# Calculate the cumulative variance
cumulative_variance_full = np.cumsum(MyPCA_full.explained_variance_ratio_)

# Find the number of components that explain at least 95% of the variance
n_components_95 = np.argmax(cumulative_variance_full >= 0.95) + 1

# Print the result
print(f"Number of components required to explain 95% of the variance: {n_components_95}")

# Get the individual explained variance and cumulative variance
explained_variance = MyPCA_full.explained_variance_ratio_
cumulative_variance_full = np.cumsum(explained_variance)

# Visualize individual explained variance with cumulative variance
plt.figure(figsize=(10, 6))
# Plot bar chart for individual explained variance
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.6, label="Individual Explained Variance",
        color='lightblue')
# Plot cumulative explained variance as a line
plt.plot(range(1, len(cumulative_variance_full) + 1), cumulative_variance_full, marker='o', linestyle='--',
         color='b', label='Cumulative Explained Variance')
# Add horizontal line at y = 0.95
plt.axhline(y=0.95, color='r', linestyle='-', label='95% Threshold')
# Adding labels and title
plt.title('Cumulative Explained Variance')
plt.xlabel('Principal Components')
plt.xlim(0)
plt.ylabel('Explained Variance')
plt.legend(loc='best')

plt.show()
