# Importación Librerias Generales
import pandas as pd
import sklearn

# Importación de Librerías Específicas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

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
    knn_class = KNeighborsClassifier().fit(x_train, y_train)
    #Predecir con valores de Test
    knn_pred = knn_class.predict(x_test)
    print('='*64)
    print('KNeighborsClassifier: ', accuracy_score(y_test, knn_pred))

    # Definir y Entrenar Clasificador Ensamblado
    bgg_class = BaggingClassifier(estimator=KNeighborsClassifier(), n_estimators=50).fit(x_train, y_train)
    #Predecir con valores de Test
    bgg_pred = knn_class.predict(x_test)
    print('='*64)
    print('BaggingClassifier-KNeighborsClassifier: ', accuracy_score(y_test, bgg_pred))

