
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd


X_train, X_test, y_train, y_test = pd.read_pickle('datos_procesados.pkl')


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)


model = LogisticRegression()
model.fit(X_train, y_train)


joblib.dump(model, 'modelo_logistico.pkl')


joblib.dump(scaler, 'scaler.pkl')

print("Modelo y escalador exportados correctamente.")
