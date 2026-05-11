from sklearn.cluster import KMeans
import numpy as np
import joblib

def train_cluster(X):
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X)
    joblib.dump(kmeans, "models/kmeans.pkl")
    return kmeans

def apply_cluster(X):
    try:
        kmeans = joblib.load("models/kmeans.pkl")
        return kmeans.predict(X)
    except:
        return np.zeros(len(X))