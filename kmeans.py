from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Cargar los datos
data = pd.read_csv('./data/Diferencia_QRESIDUOS_MUN_2014_2021.csv')

# Seleccionar las características para K-means
features = data[['QRESIDUOS_MUN_2014', 'QRESIDUOS_MUN_2021', 'QRESIDUOS_MUN_DIF']]

# Escalar los datos
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Aplicar K-means con un número determinado de clusters (por ejemplo, 3)
kmeans = KMeans(n_clusters=3, random_state=0).fit(features_scaled)

# Añadir los labels al DataFrame
data['Cluster'] = kmeans.labels_

# Visualizar los clusters
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=data['Cluster'], cmap='viridis')
plt.xlabel('QRESIDUOS_MUN_2014')
plt.ylabel('QRESIDUOS_MUN_2021')
plt.title('Clusters K-means')

plt.subplot(1, 2, 2)
plt.scatter(features_scaled[:, 0], features_scaled[:, 2], c=data['Cluster'], cmap='viridis')
plt.xlabel('QRESIDUOS_MUN_2014')
plt.ylabel('QRESIDUOS_MUN_DIF')
plt.title('Clusters K-means')

plt.show()