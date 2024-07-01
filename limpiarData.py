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
    BDgen_res.to_csv(
        './data/B_Generación_Anual_de_residuos_municipal_Distrital_2014_2021_0_modificado.csv',
        index=False,
        sep=','
    )
    BDgen_res.head()
except UnicodeDecodeError as e:
    print(f"Error al importar base de datos: {e}")

# Eliminación de filas innecesarias para el estudio
BDgen_res = BDgen_res.drop(columns=['N_SEC', 'FECHA_CORTE', 'PROVINCIA', 'DISTRITO', 'UBIGEO', 'REG_NAT'])

# Convertir 'QRESIDUOS_MUN' a float, ya que en la BD, se encuentra entre comillas.
BDgen_res['QRESIDUOS_MUN'] = BDgen_res['QRESIDUOS_MUN'].str.replace(',', '.').astype(float)

# Agrupar y sumar valores
BDgen_Depart_group = BDgen_res.groupby(['DEPARTAMENTO', 'PERIODO']).agg({ #Al agruparlo Por estos campos, los índices serán las columnas por las cuáles se agruparon.
    'POB_TOTAL': 'sum',
    'POB_URBANA': 'sum',
    'POB_RURAL': 'sum',
    'QRESIDUOS_MUN': 'sum'
}).reset_index() #Este metodo sirve para comenzar los índices de 0 y en orden creciente (numéricamente) y manejarlo de una forma más fácil.

# Encontrar los valores mínimos y máximos del campo 'PERIODO' por departamento
min_max_periods = (BDgen_Depart_group.groupby('DEPARTAMENTO')['PERIODO'].agg(['min', 'max']))

# Obtener los valores mínimos y máximos
min_period = min_max_periods['min'].min()
max_period = min_max_periods['max'].max()

# Filtrar las filas con los períodos mínimo y máximo
BDgen_Depart_group_min_max = (BDgen_Depart_group[(BDgen_Depart_group['PERIODO'] == min_period) | (BDgen_Depart_group['PERIODO'] == max_period)]).reset_index().drop(columns=['index'])

BDgen_Depart_group_min_max.to_csv('./data/B_Generación_Anual_de_residuos_municipal_Distrital_2014_2021_agrupado.csv', index=False, sep=',')

print(BDgen_Depart_group_min_max)

print("Pre procesado data 50%")