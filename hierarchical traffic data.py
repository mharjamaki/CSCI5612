import os
os.environ["OMP_NUM_THREADS"] = "1"  # Addresses memory leak warning
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns
import scipy.cluster.hierarchy as hc
from scipy.cluster.hierarchy import dendrogram


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


# Remove and save the label ############################################################################################
# Save the label
df_Label = df['DAY_OF_WEEK']
print(type(df_Label))  # Check the datatype

# Remove the label from the original dataframe
df = df.drop(['DAY_OF_WEEK'], axis=1)
print(df)  # Print the dataframe to confirm

# Save the label
df_Label2 = df['WEEKEND']
print(type(df_Label2))  # Check the datatype

# Remove the label from the original dataframe
df = df.drop(['WEEKEND'], axis=1)

# Save formatted data as csv
df.to_csv('traffic_data_clustering.csv', index=False)


# Standardize your dataset #############################################################################################
scaler = StandardScaler()  # Instantiate
df = scaler.fit_transform(df)  # Scale data
print(f"Standardized Dataframe: {df}")
print(type(df))
print(df.shape)


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

# Convert the PCA results into a DataFrame
Result_3D_df = pd.DataFrame(Result_3D)


# Hierarchical clustering ##############################################################################################
ahc = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
ahc_labels = ahc.fit_predict(Result_3D)

# Dendrogram
plt.figure(figsize=(12, 8))
plt.title("Dendrogram for 3D PCA Projection")
# Only show the top 30 clusters
dendrogram(hc.linkage(Result_3D, method='ward'), truncate_mode='lastp', p=30)
plt.xticks([])
plt.show()


# Visualize Clusters
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Result_3D[:, 0], Result_3D[:, 1], Result_3D[:, 2], c=ahc_labels, cmap='viridis', s=100, edgecolor='k')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('Agglomerative Clustering on 3D PCA Projection')
plt.show()

# Euclidean Distance Matrix
# EDist = euclidean_distances(Result_3D)
# sns.heatmap(EDist, cmap='viridis')
# plt.title('Euclidean Distance Matrix')
# plt.show()
