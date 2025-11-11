# Importación Librerias Generales
import pandas as pd
import sklearn

# Importación de Librerías Específicas
from sklearn.linear_model import (
    RANSACRegressor, HuberRegressor
)
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


if __name__ == "__main__":
    
    # Lectura del Archivo CSV
    dataset = pd.read_csv('./data_pca/felicidad_corrupt.csv')
    
    # Dividir Dataset en Features y Targets
    x = dataset.drop (['country', 'score', 'rank'], axis=1)
    y = dataset[['score']]

    # Dividir Conjunto de Entrenamiento y de Pruebas
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Generar Diccionario de Estimadores
    estimadores = {
        'SVR': SVR(gamma='auto', C=1.0, epsilon=0.1), #C y epsilon: Configuración kernel y Entrnamiento
        'RANSAC0': RANSACRegressor(), #Parámetro Opcional, si no se indica usa un Regresor Lineal (podría ser parámetro SVR)
        'HUBER': HuberRegressor(epsilon=1.35) #epsilon=1.35 teórico estístico. menor valor, menor cantidad de outliers encontrados
    }

    for name, estimador in estimadores.items():
        # Definir Estimador
        estimador.fit(x_train, y_train)
        #Predecir con valores de Test
        predictions = estimador.predict(x_test)
        print('='*64)
        print(name)
        predictions = estimador.predict(x_test)
        print('MSE: ', mean_squared_error(y_test, ))