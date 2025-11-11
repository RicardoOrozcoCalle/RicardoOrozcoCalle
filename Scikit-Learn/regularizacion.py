#Importación Librerias Generales
import pandas as pd
import sklearn

# Importación de Librerías Específicas
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    
    # Lectura del Archivo CSV
    dataset = pd.read_csv('./data_pca/felicidad.csv')

    # Dividir Dataset en Features y Targets
    x = dataset[['gdp', 'family', 'lifexp', 'freedom', 'corruption', 'generosity', 'dystopia']]
    y = dataset[['score']]

    # Dividir Conjunto de Entrenamiento y de Pruebas
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # Definir Regresión Linear
    modelLinear = LinearRegression().fit(x_train, y_train)
    y_predict_linear = modelLinear.predict(x_test)

    # Definir Regresor ElasticNet
    modelElasticNet = ElasticNet(random_state=0).fit(x_train, y_train)
    y_predict_elastic_net = modelElasticNet.predict(x_test)

    # Definir Regresor Lasso
    modelLaaso = Lasso(alpha=0.02).fit(x_train, y_train)
    y_predict_lasso = modelLaaso.predict(x_test)

    # Definir Regresor Ridge
    modelRidge = Ridge(alpha=1).fit(x_train, y_train)
    y_predict_ridge = modelRidge.predict(x_test)

    # Perdida Modelo Lineal
    linear_loss = mean_squared_error(y_test, y_predict_linear)
    print('Linear Loss:', linear_loss)

    # Perdida ElasticNet
    elastic_net_loss = mean_squared_error(y_test, y_predict_elastic_net)
    print('ElasticNet Loss:', elastic_net_loss)

    # Perdida Lasso
    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    print('Lasso Loss:', lasso_loss)
    
    # Perdida Rige
    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    print('Ridge Loss:', ridge_loss)

    print('='*64)

    # Coeficientes de Modelo Lieneal
    print('Coef Linear Model:')
    print(modelLinear.coef_)

    print('='*64)

    # Coeficientes de ElasticNet
    print('Coef ElasticNet:')
    print(modelElasticNet.coef_)

    print('='*64)

    # Coeficientes de Lasso
    print('Coef Lasso:')
    print(modelLaaso.coef_)

    print('='*64)

    # Coeficientes de Ridge
    print('Coef Ridge:')
    print(modelRidge.coef_)


    
