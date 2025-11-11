#Importación Librerias Generales
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

#Importación de Librerías Específicas
from sklearn.decomposition import KernelPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    
    #Lectura del Archivo CSV
    dt_heart = pd.read_csv('./data_pca/heart.csv')

    #Extracción de las Features del Dataset
    dt_features = dt_heart.drop(['target'], axis=1)
    #Extracción de los Targets del Dataset
    dt_target = dt_heart['target']

    #Normalización de los Datos
    dt_features = StandardScaler().fit_transform(dt_features)

    #Dividir Conjunto de Entrenamiento y de Pruebas
    X_train, X_test, Y_train, Y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42)

    
    #Configuración Algoritmo KernelPCA
    #n_components (parámetro opcional): El número de componentes es opcional, el default es:
    #n_components = min(n_muestras, n_features)    
    kpca = KernelPCA(n_components=4, kernel='poly')
    #Ajustes del KPCA a los valores de entrenamiento
    kpca.fit(X_train)

    #Configuramos los datos de entrenamiento con KernelPCA
    dt_train = kpca.transform(X_train)
    dt_test = kpca.transform(X_test)

    #Configuración del Algoritmo de Regresión Logística
    logistic = LogisticRegression(solver='lbfgs')

    # Entramos la Regresión Logística con KernelPCA
    logistic.fit(dt_train, Y_train)

    # Calculamos nuestra exactitud de nuestra predicción
    print("SCORE KernelPCA: ", logistic.score(dt_test, Y_test))

   