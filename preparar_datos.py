import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_pickle('datos.pkl')


X = df.drop(columns=['Identificación de familias beneficiarias'])  
y = df['Identificación de familias beneficiarias']  


X = pd.get_dummies(X, drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


pd.to_pickle((X_train, X_test, y_train, y_test), 'datos_procesados.pkl')
