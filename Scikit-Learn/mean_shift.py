# Importación Librerias Generales
import pandas as pd

# Importación de Librerías Específicas
from sklearn.cluster import MeanShift


if __name__ == "__main__":
    
    # Lectura del Archivo CSV
    dataset = pd.read_csv('./data_pca/candy.csv')
    
    # Dividir Dataset en Features y Targets
    x = dataset.drop (['competitorname'], axis=1)

    # Definir y Entrenar Agrupador
    meanshift = MeanShift().fit(x)
    print('Total Centros:', max(meanshift.labels_))
    print('='*64)
    print('Total Centros:', meanshift.cluster_centers_)
    print('='*64)
    dataset['meanshift'] = meanshift.labels_
    print(dataset)