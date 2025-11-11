# Importación Librerias Generales
import pandas as pd

# Importación de Librerías Específicas
from sklearn.cluster import MiniBatchKMeans


if __name__ == "__main__":
    
    # Lectura del Archivo CSV
    dataset = pd.read_csv('./data_pca/candy.csv')
    
    # Dividir Dataset en Features y Targets
    x = dataset.drop (['competitorname'], axis=1)

    # Definir y Entrenar Agrupador
    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(x)
    print('Total Centros:', len(kmeans.cluster_centers_))
    print('='*64)
    kmeans_pred = kmeans.predict(x)
    print(kmeans_pred)
    print('='*64)
    dataset['group'] = kmeans_pred
    print(dataset)