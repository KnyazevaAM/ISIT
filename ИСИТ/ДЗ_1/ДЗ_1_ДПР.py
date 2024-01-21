# ИСИТ Домашнее задание на тему: "Дерево принятия решений"
# Выполнил студент группы ББМО-01-23 Князева Анастасия Михайловна

# Задание. Необходимо, пользуясь набором данных train.csv, обучить классификатор на основе алгоритма деревьев принятия решений и проверить качество обучения. Данные для обучения представляют собой csv-таблицу, в первой колонке которой — численные значения написанных цифр, в остальных — 784 значения насыщенности отдельно взятых пикселей (картинки черно-белые).

## Загружаем необходимые библиотеки
from numpy import savetxt, loadtxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
import pandas as pd
import requests
import pickle

## Получаем файл `mnist.csv` из репозитория `GitHub`
url = "https://raw.githubusercontent.com/SergUSProject/IntelligentSystemsAndTechnologies/main/Practice/datasets/mnist.csv"
response = requests.get(url)

with open("mnist.csv", "wb") as file:
    file.write(response.content)

## Далее делим набор данных на обучающие и тестовые (70% и 30%) и загружаем обучающие данные из файла `mnist.csv` в дамп `training_set.pkl` (чтобы не хранить в оперативной памяти)
dataset = loadtxt(open('mnist.csv', 'r'), dtype='f8', delimiter=',', skiprows=1)
train1, test1 = train_test_split(dataset, test_size=0.3)
joblib.dump(train1, 'training_set.pkl')
train = joblib.load('training_set.pkl')

## Из обучающей части выделяем целевую переменную и остальную часть
target_tr = [x[0] for x in train]
train_tr = [x[1:] for x in train]

## Сохраняем тестовую выборку (без целевой переменной) в файл test_set.pkl
test_without_target = np.delete(test1, 0, axis=1)
joblib.dump(test_without_target, 'test_set.pkl')
test = joblib.load('test_set.pkl')

## Обучаем дерево принятия решений
tree = DecisionTreeClassifier()
tree.fit(train_tr, target_tr)

## Записываем результат классификации на тестовом наборе, сохраненном в виде дампа test_set.pkl, в файл answer.csv
test = joblib.load('test_set.pkl')
tree_predictions = tree.predict(test)
savetxt('answer.csv', tree_predictions, delimiter=',', fmt='%d')

## Оцените качество классификации при помощи метрики accuaracy_score
test_target = [x[0] for x in test]
accuracy = accuracy_score(test_target, tree_predictions)
print("Accuracy:", accuracy)