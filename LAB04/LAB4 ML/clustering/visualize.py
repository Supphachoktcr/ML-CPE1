import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def plot_elbow_and_clusters(X, df_original, labels):
    os.makedirs('outputs', exist_ok=True)
    
    # 1. Elbow Method Graph
    inertia = []
    K_range = range(1, 10)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertia.append(km.inertia_)
        
    plt.figure(figsize=(8, 4))
    plt.plot(K_range, inertia, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.savefig('outputs/01_elbow.png')
    plt.close()

    # 2. PCA Clusters Plot (2D)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50)
    plt.title('Animal Clusters Visualization (PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig('outputs/02_clusters.png')
    plt.close()