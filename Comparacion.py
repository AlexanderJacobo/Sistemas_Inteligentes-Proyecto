import csv
import pandas as pd
import numpy as np
import io
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.base import TransformerMixin
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

#Importar base de datos.
try:
    BDgen_res = pd.read_csv(
        './data/B_Generación_Anual_de_residuos_municipal_Distrital_2014_2021_0.csv', 
        delimiter=';', #En la base de datos el delimitadore es ";"
        encoding='ISO-8859-1' #No funciona con UTF-8
    )
    BDgen_res.head()
except UnicodeDecodeError as e:
    print(f"Error al importar base de datos: {e}")

# Eliminación de filas innecesarias para el estudio
BDgen_res = BDgen_res.drop(columns=['N_SEC', 'FECHA_CORTE', 'PROVINCIA', 'DISTRITO', 'UBIGEO', 'POB_URBANA','REG_NAT','POB_RURAL'])

# Convertir 'QRESIDUOS_MUN' a float, ya que en la BD, se encuentra entre comillas.
BDgen_res['QRESIDUOS_MUN'] = BDgen_res['QRESIDUOS_MUN'].str.replace(',', '.').astype(float)

# Agrupar y sumar valores
BDgen_Depart_group = BDgen_res.groupby(['DEPARTAMENTO', 'PERIODO']).agg({ #Al agruparlo Por estos campos, los índices serán las columnas por las cuáles se agruparon.
    'POB_TOTAL': 'sum',
    'QRESIDUOS_MUN': 'sum'
}).reset_index() #Este metodo sirve para comenzar los índices de 0 y en orden creciente (numéricamente) y manejarlo de una forma más fácil.

# Encontrar los valores mínimos y máximos del campo 'PERIODO' por departamento
min_max_periods = (BDgen_Depart_group.groupby('DEPARTAMENTO')['PERIODO'].agg(['min', 'max']))

# Obtener los valores mínimos y máximos
min_period = min_max_periods['min'].min()
max_period = min_max_periods['max'].max()

# Filtrar las filas con los períodos mínimo y máximo
BDgen_Depart_Filtered = (BDgen_Depart_group[(BDgen_Depart_group['PERIODO'] == min_period) | (BDgen_Depart_group['PERIODO'] == max_period)]).reset_index().drop(columns=['index'])

BDgen_Depart_Filtered.to_csv('./data/Generación_Anual_de_Residuos_Municipal_Distrital_2014_2021.csv', index=False, sep=',')

# Crear nuevos campos basados en intervalos de millones para POB_TOTAL
min_pob = BDgen_Depart_Filtered['POB_TOTAL'].min() // 1000000 * 1000000
max_pob = (BDgen_Depart_Filtered['POB_TOTAL'].max() // 1000000 + 1) * 1000000
pob_intervals = range(int(min_pob), int(max_pob), 1000000)

# Crear nuevos campos basados en intervalos de 50,000 para QRESIDUOS_MUN
min_res = BDgen_Depart_Filtered['QRESIDUOS_MUN'].min() // 50000 * 50000
max_res = (BDgen_Depart_Filtered['QRESIDUOS_MUN'].max() // 50000 + 1) * 50000
res_intervals = range(int(min_res), int(max_res), 50000)

# Obtener valores únicos de DEPARTAMENTO y PERIODO
departamentos = BDgen_Depart_Filtered['DEPARTAMENTO'].unique()
periodos = BDgen_Depart_Filtered['PERIODO'].unique()

# Crear la nueva tabla
columns = list(departamentos) + list(periodos) + \
          [f'Pob_{i//1000000}M_{(i+1000000)//1000000}M' for i in pob_intervals] + \
          [f'Residuos_{i//1000}k_{(i+50000)//1000}k' for i in res_intervals]

BDgen_preprocesada = pd.DataFrame(columns=columns)

# Llenar la nueva tabla con los registros de la tabla anterior
for _, row in BDgen_Depart_Filtered.iterrows():
    new_row = {col: 'FALSE' for col in columns}
    
    # Actualizar columnas de DEPARTAMENTO
    if row['DEPARTAMENTO'] in new_row:
        new_row[row['DEPARTAMENTO']] = 'TRUE'
    
    # Actualizar columnas de PERIODO
    for periodo in periodos:
        new_row[periodo] = 1 if row['PERIODO'] == periodo else 0
    
    # Actualizar columnas de POB_TOTAL
    for i in pob_intervals:
        col_name = f'Pob_{i//1000000}M_{(i+1000000)//1000000}M'
        new_row[col_name] = 'TRUE' if i <= row['POB_TOTAL'] < i + 1000000 else 'FALSE'
    
    # Actualizar columnas de QRESIDUOS_MUN
    for i in res_intervals:
        col_name = f'Residuos_{i//1000}k_{(i+50000)//1000}k'
        new_row[col_name] = 1 if i <= row['QRESIDUOS_MUN'] < i + 50000 else 0
    
    BDgen_preprocesada = BDgen_preprocesada._append(new_row, ignore_index=True)

# Eliminar columnas con todos los valores FALSE/0
columns_to_keep = BDgen_preprocesada.columns[(BDgen_preprocesada != 'FALSE').any(axis=0)]
columns_to_keep = columns_to_keep[(BDgen_preprocesada[columns_to_keep] != 0).any(axis=0)]
BDgen_preprocesada = BDgen_preprocesada[columns_to_keep]

BDgen_preprocesada.to_csv('./data/Generación_Anual_de_Residuos_Municipal_Distrital_Procesado.csv', index=False, sep=',')

# Separar los datos de 2014 y 2021
data_2014 = BDgen_Depart_Filtered[BDgen_Depart_Filtered['PERIODO'] == 2014].reset_index(drop=True)
data_2021 = BDgen_Depart_Filtered[BDgen_Depart_Filtered['PERIODO'] == 2021].reset_index(drop=True)

# Combinar los datos basados en el departamento
combined_data = pd.merge(data_2014, data_2021, on='DEPARTAMENTO', suffixes=('_2014', '_2021'))

# Calcular la diferencia de QRESIDUOS_MUN entre 2021 y 2014
combined_data['QRESIDUOS_MUN_DIF'] = combined_data['QRESIDUOS_MUN_2021'] - combined_data['QRESIDUOS_MUN_2014']

# Seleccionar las columnas que queremos mostrar
result = combined_data[['DEPARTAMENTO', 'QRESIDUOS_MUN_2014', 'QRESIDUOS_MUN_2021', 'QRESIDUOS_MUN_DIF']]

# Guardar el resultado en un nuevo archivo CSV
result.to_csv('./data/Diferencia_QRESIDUOS_MUN_2014_2021.csv', index=False, sep=',')

print(result)
