import pandas as pd
from data_loader import load_data_for_clustering
from kmeans_tf import fit_kmeans
from visualize import plot_elbow_and_clusters

def main():
    data_path = '../data-animal/animal_dataset.csv'
    df_original, X_scaled = load_data_for_clustering(data_path)
    
    # กำหนด n_clusters = 7 ตามประเภทหลักของ Zoo Dataset
    n_clusters = 7
    kmeans_model, labels = fit_kmeans(X_scaled, n_clusters=n_clusters)
    
    plot_elbow_and_clusters(X_scaled, df_original, labels)
    
    df_original['Cluster'] = labels
    df_original.to_csv('outputs/clustered_animals.csv', index=False)
    
    summary = df_original.groupby('Cluster').mean(numeric_only=True)
    summary.to_csv('outputs/cluster_summary.csv')
    
    print("Clustering completed. Outputs saved in clustering/outputs/")

if __name__ == "__main__":
    main()