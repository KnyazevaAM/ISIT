# ИСИТ Домашнее задание на тему: "Линейная регрессия"
# Выполнил студент группы ББМО-01-23 Князева Анастасия Михайловна

## 1. Импортируем нужные для работы программы, а также импортируем из репозитория гитхаб нужный нам файл `Davis.csv`:
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import requests

url = "https://raw.githubusercontent.com/sdukshis/ml-intro/master/datasets/Davis.csv"
response = requests.get(url)

with open("Davis.csv", "wb") as file:
    file.write(response.content)

## 2. Подготовим наш CSV файл для работы, а именно для правильной работы с данными поля `sex` нужно поменять ему тип переменной с `string` на `int`. Для этого сменим в каждом поле столбца значения с `F` на `1` и с `M` на `0` соответственно:
# Откройте CSV-файл
df = pd.read_csv('Davis.csv')

# Произведите замену
df.replace({'sex': {'F': 1, 'M': 0}}, inplace=True)

# Запишите измененный DataFrame обратно в CSV-файл
df.to_csv('Davis.csv',index=False)

## 3. Загрузим данные об измерениях роста и веса из CSV файла `Davis.csv`:
data = pd.read_csv('Davis.csv')

## 4. Просмотрим записи в загруженном файле:
print(data.head())

## 5. Уберем из данных пропуски при помощи метода `pandas.Dataframe.dropna`:
data = data.dropna()

## 6. Создадим переменные 'X' и 'Y', где 'X' будет целевой переменной и содержать значения роста, а 'Y' - переменная, содержащая признаки - значение веса:
X = data[['height']].values
Y = data['weight'].values

## 7. Далее создадим объект `Regressor` и обучим его на полученных данных:
Regressor = LinearRegression()
Regressor.fit(X, Y)

## 8. После обучения модели линейной регрессии можно использовать её для предсказания веса на основе роста:
new_height = [[180]]
predicted_weight = Regressor.predict(new_height)
rounded_weight = round(predicted_weight[0])
print("Predicted weight:", rounded_weight)

## 9. Вычислим значение среднеквадратичной ошибки для построенной модели путем импортирования функции `mean_squared_error` из библиотеки `scikit-learn` и выолним расчет среднеквадратичной ошибки для тестовой выборки:
Y_pred = Regressor.predict(X)
mse = mean_squared_error(Y, Y_pred)
rounded_mse = np.round(mse)
print('Среднеквадратичная ошибка:', rounded_mse)

## 10. Построим прямую регрессии и проиллюстрируем точки обучающей выборки при помощи библиотеки `matplotlib`:
plt.scatter(X, Y, color='blue', label='Обучающая выборка')
plt.plot(X, Y_pred, color='red', linewidth=1, label='Прямая регрессии')
plt.xlabel('weight')
plt.ylabel('height')
plt.legend()
plt.show()

## 11. Расширим количество признаков, добавив пол и repwt путем загрузки из CSV файла и объединения с уже имеющимся признаком веса:
### 11.1 Создадим переменные 'X' и 'Y', где 'X' будет целевой переменной и содержать значения роста, а 'Y' - переменная, содержащая признаки - значение веса и добавленные занчения пола и repwt:
X = data['height'].values.reshape(-1, 1)
Y = data[['weight', 'sex', 'repwt']].values

### 11.2 Создадим объект `Model` и обучим его на дополненных данных:
Model = LinearRegression()
Model.fit(X, Y)

### 11.3 Вычислим значение среднеквадратичной ошибки для построенной модели путем импортирования функции mean_squared_error из библиотеки scikit-learn и выолним расчет среднеквадратичной ошибки для тестовой выборки:
Y_pred = Model.predict(X)
mse = mean_squared_error(Y, Y_pred)
rounded_mse = np.round(mse)
print('Среднеквадратичная ошибка:', rounded_mse)


