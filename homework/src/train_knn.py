#
# Busque los mejores parametros de un modelo knn para predecir
# la calidad del vino usando el dataset de calidad del vino tinto de UCI.
#
# Considere diferentes valores para la cantidad de vecinos
#

# importacion de librerias
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

from homework.internals.calculate_metrics import calculate_metrics
from homework.internals.prepare_data import prepare_data
from homework.internals.print_metrics import print_metrics
from homework.internals.save_model_if_better import save_model_if_better

# dividir los datos en entrenamiento y testing
x_train, x_test, y_train, y_test = prepare_data(
    file_path="data/",
    test_size=0.25,
    random_state=123456
)

# entrenar el modelo
estimator = KNeighborsRegressor(n_neighbors=5)
estimator.fit(x_train, y_train)

# Metricas de error durante entrenamiento
mse, mae, r2 = calculate_metrics(estimator, x_train, y_train)
print_metrics("Training metrics", mse, mae, r2)

# Metricas de error durante testing
mse, mae, r2 = calculate_metrics(estimator, x_test, y_test)
print_metrics("Metricas de testing", mse, mae, r2)

save_model_if_better(estimator, x_test, y_test)
