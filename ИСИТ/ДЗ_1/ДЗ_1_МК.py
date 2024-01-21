# ИСИТ Домашнее задание на тему: "Метрики классификации"
# Выполнил студент группы ББМО-01-23 Князева Анастасия Михайловна

# В данном практическом задании предлагается решить задачу бинарной классификации с помощью 2-х методов:

# *   логистической регрессии;
# *   метода k ближайших соседей.

## Необходимые библиотеки:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt

## Сгенерированный набор данных для задач бинарной классификации:
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples = 1000,
    n_features = 2,
    n_informative = 2,
    n_redundant = 0,
    n_repeated = 0,
    n_classes = 2,
    n_clusters_per_class = 1,
    weights = (0.15, 0.85),
    class_sep = 6.0,
    hypercube = False,
    random_state = 2,
)

## Разделение выборки на обучающую и проверочную: 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Реализация метода логисчтической регрессии:
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

##  Реализация метода k ближайших соседей:
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

## Рассчитанные метрики для логистической регрессии:
print("Логистическая регрессия:")
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_conf_matrix = confusion_matrix(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)
lr_avg_precision = average_precision_score(y_test, lr_pred)
lr_roc_auc = roc_auc_score(y_test, lr_pred)
print(f"Доля верных ответов: {lr_accuracy}")
print(f"Матрица ошибок:\n{lr_conf_matrix}")
print(f"Точность: {lr_precision}")
print(f"Полнота: {lr_recall}")
print(f"F-мера: {lr_f1}")
print(f"Average Precision: {lr_avg_precision}")
print(f"ROC-AUC: {lr_roc_auc}")

## Рассчитанные метрики для метода k ближайших соседей:
print("\nМетод k ближайших соседей:")
knn_accuracy = accuracy_score(y_test, knn_pred)
knn_conf_matrix = confusion_matrix(y_test, knn_pred)
knn_precision = precision_score(y_test, knn_pred)
knn_recall = recall_score(y_test, knn_pred)
knn_f1 = f1_score(y_test, knn_pred)
knn_avg_precision = average_precision_score(y_test, knn_pred)
knn_roc_auc = roc_auc_score(y_test, knn_pred)

print(f"Доля верных ответов: {knn_accuracy}")
print(f"Матрица ошибок:\n{knn_conf_matrix}")
print(f"Точность: {knn_precision}")
print(f"Полнота: {knn_recall}")
print(f"F-мера: {knn_f1}")
print(f"Average Precision: {knn_avg_precision}")
print(f"ROC-AUC: {knn_roc_auc}")

## Визуализация PR-Кривых:
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr.decision_function(X_test))
knn_precision, knn_recall, _ = precision_recall_curve(y_test, knn.predict_proba(X_test)[:, 1])

plt.figure(figsize=(10, 6))
plt.plot(lr_recall, lr_precision, label=f'Логистическая регрессия (AP={lr_avg_precision:.2f})')
plt.plot(knn_recall, knn_precision, label=f'k ближайших соседей (AP={knn_avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR-кривые')
plt.legend()
plt.show()

## Визуализация ROC-кривых:
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr.decision_function(X_test))
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn.predict_proba(X_test)[:, 1])

plt.figure(figsize=(10, 6))
plt.plot(lr_fpr, lr_tpr, label=f'Логистическая регрессия (AUC={lr_roc_auc:.2f})')
plt.plot(knn_fpr, knn_tpr, label=f'k ближайших соседей (AUC={knn_roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривые')
plt.legend()
plt.show()



