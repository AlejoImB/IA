import pandas as pd


with open('precision.txt', 'r') as f:
    precision = f.read()


respuestas = f"""
Precisión del modelo: {precision}
Variables de entrenamiento: Son las columnas del conjunto de datos procesado.
Variable predicha: 'Identificación de familias beneficiarias'
Hiperparámetros del modelo: Los valores predeterminados de LogisticRegression
"""


with open('respuestas.txt', 'w') as f:
    f.write(respuestas)

print("Respuestas guardadas en 'respuestas.txt'")
