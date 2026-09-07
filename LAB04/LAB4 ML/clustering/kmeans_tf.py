from sklearn.cluster import KMeans

def fit_kmeans(X, n_clusters=7):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    return kmeans, labels