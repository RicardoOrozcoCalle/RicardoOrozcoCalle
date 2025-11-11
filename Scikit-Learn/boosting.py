# Importación Librerias Generales
import pandas as pd
import sklearn

# Importación de Librerías Específicas
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    
    # Lectura del Archivo CSV
    dt_heart = pd.read_csv('./data_pca/heart.csv')
    print(dt_heart.describe())
    
    # Dividir Dataset en Features y Targets
    x = dt_heart.drop (['target'], axis=1)
    y = dt_heart[['target']]

    # Dividir Conjunto de Entrenamiento y de Pruebas
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35)

    # Definir y Entrenar Clasificador
    boost = GradientBoostingClassifier(n_estimators=50).fit(x_train, y_train)
    #Predecir con valores de Test
    boost_pred = boost.predict(x_test)
    print('='*64)
    print('GradientBoostingClassifier: ', accuracy_score(y_test, boost_pred))

