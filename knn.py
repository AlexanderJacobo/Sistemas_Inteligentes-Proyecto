import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# Cargar los datos
data = pd.read_csv('./data/Diferencia_QRESIDUOS_MUN_2014_2021.csv')

# Crear etiquetas basadas en QRESIDUOS_MUN_DIF
def categorize_diff(value):
    if value < 0:
        return 0  
    elif value == 564899.52:  
        return 3  # Lima
    elif value > 30000:
        return 2  
    else:
        return 1  

# Aplicar la función categorize_diff para crear la nueva columna 'Class'
data['Class'] = data['QRESIDUOS_MUN_DIF'].apply(categorize_diff)

# Seleccionar las características y la nueva etiqueta
X = data[['QRESIDUOS_MUN_2014', 'QRESIDUOS_MUN_2021']].values
y = data['Class'].values

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Entrenar el modelo KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predicciones del modelo KNN
y_pred = knn.predict(X_test)

# Evaluar el rendimiento del modelo
print("Accuracy:", knn.score(X_test, y_test))

# Visualización de la clasificación
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#A90EF1'])

x_min, x_max = X_scaled[:, 0].min() - .1, X_scaled[:, 0].max() + .1
y_min, y_max = X_scaled[:, 1].min() - .1, X_scaled[:, 1].max() + .1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure()

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('QRESIDUOS_MUN_2014')
plt.ylabel('QRESIDUOS_MUN_2021')
plt.title('Clasificación KNN')
plt.show()
