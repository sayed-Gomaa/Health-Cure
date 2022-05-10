
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph
import seaborn as sns # used for plot interactive graph. I like it most for plot
from sklearn.linear_model import LogisticRegression # to apply the Logistic regression
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.model_selection import KFold # use for cross validation
from sklearn.model_selection import GridSearchCV# for tuning parameter
from sklearn.ensemble import RandomForestClassifier # for random forest classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm # for Support Vector Machine
from sklearn import metrics # for the check the error and accuracy of the model
import joblib
# Any results you write to the current directory are saved as output.
# dont worry about the error if its not working then insteda of model_selection we can use cross_validation


data = pd.read_csv("data.csv",header=0)# here header 0 means the 0 th row is our coloumn header in data
#now split our data into train and test
train, test = train_test_split(data, test_size = 0.3) # in this our main data is splitted into train and test

features= list(data.columns[2:-1])
prediction_var = features
train_X= train[prediction_var]
train_y= train.diagnosis
test_X = test[prediction_var]
test_y = test.diagnosis

model = RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_y)
prediction = model.predict(test_X)
metrics.accuracy_score(prediction,test_y)

print('Accuracy : {}'.format(metrics.accuracy_score(prediction, test_y)))

clf_report = metrics.classification_report(prediction, test_y)
print('Classification report')
print("---------------------")
print(clf_report)
print("_____________________")

joblib.dump(model,"cancer_model.pkl")
