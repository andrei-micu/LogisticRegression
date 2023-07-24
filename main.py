from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from logistic_regression import LogisticRegression as CustomLogisticRegression
from data import x_train, x_test, y_train, y_test

import numpy as np

my_matrix1 = np.array([[1, 2, 3], [4, 5, 6]])
my_matrix2 = np.array([[7, 8, 9], [10, 11, 12]])

for elemtn in my_matrix1:
    print(elemtn)


lr = CustomLogisticRegression()
lr.fit(x_train, y_train, epochs=150)
pred = lr.predict(x_test)
accuracy = accuracy_score(y_test, pred)
print(accuracy)
print(lr.losses)
print(lr.train_accuracies)

model = LogisticRegression(solver='newton-cg', max_iter=150)
model.fit(x_train, y_train)
pred2 = model.predict(x_test)
accuracy2 = accuracy_score(y_test, pred2)
print(accuracy2)
