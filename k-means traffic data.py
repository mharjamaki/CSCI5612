import os
os.environ["OMP_NUM_THREADS"] = "1"  # Addresses memory leak warning
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


# Load data ############################################################################################################
path = "C:/Users/mikyl/OneDrive/Documents/CSCI/Cleaned Data/traffic_cleaned.csv"
df = pd.read_csv(path)


# Format data ##########################################################################################################
df.drop(columns=['COUNTDATE'], inplace=True)
df.reset_index(drop=True)


# Remove and save the label ############################################################################################
df_Label = df['DAY_OF_WEEK']
df_Label2 = df['WEEKEND']


# Remove labels from original dataframe
df = df.drop(['DAY_OF_WEEK', 'WEEKEND'], axis=1)


# Standardize your dataset #############################################################################################
scaler = StandardScaler()  # Instantiate
df = scaler.fit_transform(df)  # Scale data


# Perform 3D PCA #######################################################################################################
MyPCA_3D = PCA(n_components=3)
Result_3D = MyPCA_3D.fit_transform(df)

# Convert the PCA results into a DataFrame
Result_3D_df = pd.DataFrame(Result_3D)


# Selecting K with silhouette ##########################################################################################
range_n_clusters = [2, 3, 4, 5, 6, 7]

# Track silhouette scores
silhouette_scores = []

for n_clusters in range_n_clusters:
    # Initialize KMeans clustering
    clusterer = KMeans(n_clusters=n_clusters, random_state=10, n_init=10)
    cluster_labels = clusterer.fit_predict(Result_3D)

    # Calculate silhouette score
    silhouette_avg = silhouette_score(Result_3D, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot silhouette score vs. number of clusters
plt.figure(figsize=(8, 6))
plt.plot(range_n_clusters, silhouette_scores, marker='o', linestyle='-', color='b')
plt.title("Silhouette Score vs. Number of Clusters")
plt.xlabel("Number of Clusters")
plt.xlim(2)
plt.ylabel("Average Silhouette Score")
plt.grid(False)

# Find index of the maximum silhouette score
best_n_clusters = range_n_clusters[np.argmax(silhouette_scores)]

# Add a dashed vertical line at the maximum silhouette score
plt.axvline(x=best_n_clusters, color='r', linestyle='--', label=f"Peak at {best_n_clusters} clusters")
plt.xticks(np.arange(min(range_n_clusters), max(range_n_clusters) + 1, 1))
plt.legend()
plt.show()


# Perform K-Means Clustering ###########################################################################################
# Perform K-means clustering on the PCA results
kmeans = KMeans(n_clusters=2, n_init=10)  # Define the number of clusters
kmeans.fit(Result_3D)

# Get the cluster labels assigned to each data point
labels = kmeans.labels_

# Add the cluster labels to the Result_3D DataFrame
Result_3D_df['Cluster'] = labels


# Visualize the clusters in 3D #########################################################################################
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(Result_3D[:, 0], Result_3D[:, 1], Result_3D[:, 2], c=labels, cmap='spring', s=75, edgecolor='k')
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
ax.set_title('K-means Clusters on 3D PCA Projection')

# Plotting the centroids on the 3D plot
centers3D = kmeans.cluster_centers_
print(centers3D)

# Scatter plot of centroids
ax.scatter(centers3D[:, 0], centers3D[:, 1], centers3D[:, 2], c='black', s=900, alpha=0.95, marker='o', edgecolor='k',
           linewidth=2)

plt.show()


# Visualizing the clusters in 3D with true labels ######################################################################
fig4 = plt.figure()
ax4 = fig4.add_subplot(projection='3d')
x = Result_3D[:, 0]
y = Result_3D[:, 1]
z = Result_3D[:, 2]
unique_labels2 = np.unique(df_Label2)
cmap = plt.get_cmap("Set1", len(unique_labels2))
norm = colors.BoundaryNorm(np.arange(len(unique_labels2) + 1), cmap.N)
sc3d2 = ax4.scatter(x, y, z, cmap=cmap, edgecolor='k', s=200, c=df_Label2, norm=norm)
ax4.set_xlabel('PCA 1')
ax4.set_ylabel('PCA 2')
ax4.set_zlabel('PCA 3')
ax4.set_title('Weekday/Weekend Labels on 3D PCA Projection')
plt.colorbar(sc3d2, label='Weekday/Weekend', ticks=np.arange(len(unique_labels2)))

plt.show()
