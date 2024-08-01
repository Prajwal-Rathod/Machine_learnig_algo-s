import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Encode categorical columns
    labelencoder = LabelEncoder()
    data['Gender'] = labelencoder.fit_transform(data['Gender'])
    return data

def k_means_clustering(data, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(data)
    labels = model.predict(data)
    score = silhouette_score(data, labels)
    return labels, score

def expectation_maximization(data, n_components):
    model = GaussianMixture(n_components=n_components, random_state=42)
    model.fit(data)
    labels = model.predict(data)
    score = silhouette_score(data, labels)
    return labels, score

# Example usage:
data = read_csv('7ds.csv')
data = preprocess_data(data)

# Using only numerical columns for clustering
data_for_clustering = data[['Gender', 'Age', 'AnnualIncome (k$)', 'SpendingScore (1-100)']]

kmeans_labels, kmeans_score = k_means_clustering(data_for_clustering, 3)
em_labels, em_score = expectation_maximization(data_for_clustering, 3)

print("K-Means Clustering Score:", kmeans_score)
print("Expectation Maximization Score:", em_score)
