import os
os.environ["OMP_NUM_THREADS"] = "1"  # Addresses memory leak warning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib import colors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import DBSCAN
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

# Save the label
df_Label2 = df['WEEKEND']
print(type(df_Label2))  # Check the datatype

# Remove the label from the original dataframe
df = df.drop(['WEEKEND'], axis=1)


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


# DBSCAN ###############################################################################################################
db = DBSCAN(eps=2, min_samples=10)
# Eps: the maximum distance between two samples for one to be considered as in the neighborhood of the other
# Min_samples: the number of samples in a neighborhood for a point to be considered as a core point. This includes the
# point itself. If it is set to a higher value, DBSCAN will find denser clusters
db.fit_predict(Result_3D)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

unique_labels = set(labels)
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

# Create a 3D plot for DBSCAN clustering result
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    # Core samples
    xy = Result_3D[class_member_mask & core_samples_mask]
    ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], c=[col], edgecolor='k', s=50, marker='o')

    # Non-core samples
    xy = Result_3D[class_member_mask & ~core_samples_mask]
    ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], c=[col], edgecolor='k', s=30, marker='o')

ax.set_title(f"DBSCAN Clustering (Estimated {n_clusters_} clusters, {n_noise_} noise points)")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")

plt.show()
