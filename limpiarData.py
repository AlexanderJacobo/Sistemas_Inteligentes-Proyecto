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
    BDgen_res = pd.read_csv('./data/B_Generación_Anual_de_residuos_municipal_Distrital_2014_2021_0.csv', encoding='ISO-8859-1')
    print(BDgen_res)
    BDgen_res.head()
except UnicodeDecodeError as e:
    print(f"Error decoding file: {e}")
