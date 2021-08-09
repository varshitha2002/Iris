from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
#from sklearn.neighbors import KNeighborsClassifier
import joblib
import numpy as np
import csv

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

logreg = LogisticRegression(solver='lbfgs',max_iter = 1000)
logreg.fit(X_train, y_train)
joblib.dump(logreg, 'iris-predictor.joblib')
model = joblib.load('iris-predictor.joblib')

### Testing the accuracy using the test data
#y_pred = model.predict(X_test)
#print(y_pred,y_test)
#score = metrics.accuracy_score(y_test,y_pred)
#score

with open("test.csv") as file_name:
    file_read = csv.reader(file_name, delimiter = ",")

    array = list(file_read)
array.pop(0)
for i in range(len(array)):
    array[i].pop(0)
    array[i] = list(map(int,array[i]))

header = ['ID', 'SepalLength', 'SepalWidth', 'Petallength', 'PetalWidth', 'Prediction']

with open('Results.csv', 'w') as fp:
    writer = csv.writer(fp)
    writer.writerow(header)
    for i in range(len(array)):
        prediction_Array = []
        prediction_Array.append(str(i+1))
        prediction = iris.target_names[model.predict([array[i]])[0]]
        for values in array[i]:
            prediction_Array.append(values)
        prediction_Array.append(prediction.capitalize())
        writer.writerow(prediction_Array)
        print(f"{i+1}) {prediction}")
fp.close()