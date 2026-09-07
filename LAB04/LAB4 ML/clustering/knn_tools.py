from sklearn.neighbors import NearestNeighbors

def find_nearest_neighbors(X, n_neighbors=3):
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    return distances, indices