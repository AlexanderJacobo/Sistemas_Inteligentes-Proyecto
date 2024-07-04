import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Cargar los datos
data = pd.read_csv('./data/Diferencia_QRESIDUOS_MUN_2014_2021.csv')

# Seleccionar las características para K-means
features = data[['QRESIDUOS_MUN_2014', 'QRESIDUOS_MUN_2021', 'QRESIDUOS_MUN_DIF']]

# Escalar los datos
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Evaluar el índice de Silhouette para diferentes valores de n_clusters
silhouette_scores = []
for n_clusters in range(2, 11):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = model.fit_predict(features_scaled)
    silhouette_avg = silhouette_score(features_scaled, cluster_labels)
    silhouette_scores.append((n_clusters, silhouette_avg))
    print(f"Para n_clusters = {n_clusters}, el índice de Silhouette es {silhouette_avg}")

# Determinar el mejor valor de n_clusters según el índice de Silhouette
best_n_clusters = max(silhouette_scores, key=lambda item: item[1])[0]
print(f"El mejor valor de n_clusters según el índice de Silhouette es {best_n_clusters}")

# Aplicar K-means con el mejor número de clusters
kmeans = KMeans(n_clusters=best_n_clusters, random_state=42).fit(features_scaled)

# Añadir los labels al DataFrame
data['Cluster'] = kmeans.labels_

# Visualizar los clusters
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=data['Cluster'], cmap='viridis')
plt.xlabel('QRESIDUOS_MUN_2014')
plt.ylabel('QRESIDUOS_MUN_2021')
plt.title(f'Clusters K-means (n_clusters={best_n_clusters})')

plt.subplot(1, 2, 2)
plt.scatter(features_scaled[:, 0], features_scaled[:, 2], c=data['Cluster'], cmap='viridis')
plt.xlabel('QRESIDUOS_MUN_2014')
plt.ylabel('QRESIDUOS_MUN_DIF')
plt.title(f'Clusters K-means (n_clusters={best_n_clusters})')

plt.show()

# Visualización de la evaluación del índice de Silhouette
n_clusters_list, silhouette_avg_list = zip(*silhouette_scores)
plt.plot(n_clusters_list, silhouette_avg_list, marker='o')
plt.xlabel('Número de clusters (n_clusters)')
plt.ylabel('Índice de Silhouette')
plt.title('Evaluación del índice de Silhouette para diferentes valores de n_clusters')
plt.show()
