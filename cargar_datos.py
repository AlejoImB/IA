import pandas as pd

df = pd.read_csv('familias_accion.csv')


df.to_pickle('datos.pkl')


print(df.head())
