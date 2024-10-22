import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


X_train, X_test, y_train, y_test = pd.read_pickle('datos_procesados.pkl')


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LogisticRegression()
model.fit(X_train, y_train)


accuracy = model.score(X_test, y_test)
print(f'Precisión del modelo: {accuracy}')


with open('precision.txt', 'w') as f:
    f.write(str(accuracy))
