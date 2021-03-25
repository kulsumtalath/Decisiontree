```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import warnings  
warnings.filterwarnings('ignore')
```


```python
def importdata():
    balance_data=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-'+
    'databases/balance-scale/balance-scale.data',sep=',',header=None)
    print("Dataset Length",len(balance_data))
    print("Dataset shape",balance_data.shape)
    print("Dataset: ",balance_data.head())
    return balance_data
```


```python
def splitdataset(balance_data):
    X=balance_data.values[:,1:5]
    Y=balance_data.values[:,0]
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=100)
    return X,Y,X_train,X_test,Y_train,Y_test
```


```python
def train_using_gini(X_train,X_test,Y_train):
    clf_gini=DecisionTreeClassifier(criterion="gini",random_state=100,max_depth=3,min_samples_leaf=5)
    clf_gini.fit(X_train,Y_train)
    return clf_gini
    
```


```python
def train_using_entropy(X_train,X_test,Y_train):
    clf_entropy=DecisionTreeClassifier(criterion="entropy",random_state=100,max_depth=3,min_samples_leaf=5)
    clf_entropy.fit(X_train,Y_train)
    return clf_entropy
    
    
```


```python
def prediction(X_test,clf_object):
    Y_pred=clf_object.predict(X_test)
    print("Predcited values:")
    print(Y_pred)
    return Y_pred
    
```


```python
def cal_accuracy(Y_test,Y_pred):
    print("Confusion Matrix: ",confusion_matrix(Y_test,Y_pred))
    print("Accuracy: ",accuracy_score(Y_test,Y_pred)*100)
    print("Report: ",classification_report(Y_test,Y_pred))
```


```python
def main():
    data=importdata()
    X,Y,X_train,X_test,Y_train,Y_test=splitdataset(data)
    clf_gini = train_using_gini(X_train, X_test, Y_train)
    clf_entropy = train_using_entropy(X_train, X_test, Y_train)
    print("Results using Gini Index: ")
    Y_pred_gini=prediction(X_test,clf_gini)
    cal_accuracy(Y_test,Y_pred_gini)
    print("Results Using Entropy:")
    # Prediction using entropy
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(Y_test, y_pred_entropy)
                              
if __name__=="__main__":
    main()                           
```

    Dataset Length 625
    Dataset shape (625, 5)
    Dataset:     0  1  2  3  4
    0  B  1  1  1  1
    1  R  1  1  1  2
    2  R  1  1  1  3
    3  R  1  1  1  4
    4  R  1  1  1  5
    Results using Gini Index: 
    Predcited values:
    ['R' 'L' 'R' 'R' 'R' 'L' 'R' 'L' 'L' 'L' 'R' 'L' 'L' 'L' 'R' 'L' 'R' 'L'
     'L' 'R' 'L' 'R' 'L' 'L' 'R' 'L' 'L' 'L' 'R' 'L' 'L' 'L' 'R' 'L' 'L' 'L'
     'L' 'R' 'L' 'L' 'R' 'L' 'R' 'L' 'R' 'R' 'L' 'L' 'R' 'L' 'R' 'R' 'L' 'R'
     'R' 'L' 'R' 'R' 'L' 'L' 'R' 'R' 'L' 'L' 'L' 'L' 'L' 'R' 'R' 'L' 'L' 'R'
     'R' 'L' 'R' 'L' 'R' 'R' 'R' 'L' 'R' 'L' 'L' 'L' 'L' 'R' 'R' 'L' 'R' 'L'
     'R' 'R' 'L' 'L' 'L' 'R' 'R' 'L' 'L' 'L' 'R' 'L' 'R' 'R' 'R' 'R' 'R' 'R'
     'R' 'L' 'R' 'L' 'R' 'R' 'L' 'R' 'R' 'R' 'R' 'R' 'L' 'R' 'L' 'L' 'L' 'L'
     'L' 'L' 'L' 'R' 'R' 'R' 'R' 'L' 'R' 'R' 'R' 'L' 'L' 'R' 'L' 'R' 'L' 'R'
     'L' 'L' 'R' 'L' 'L' 'R' 'L' 'R' 'L' 'R' 'R' 'R' 'L' 'R' 'R' 'R' 'R' 'R'
     'L' 'L' 'R' 'R' 'R' 'R' 'L' 'R' 'R' 'R' 'L' 'R' 'L' 'L' 'L' 'L' 'R' 'R'
     'L' 'R' 'R' 'L' 'L' 'R' 'R' 'R']
    Confusion Matrix:  [[ 0  6  7]
     [ 0 67 18]
     [ 0 19 71]]
    Accuracy:  73.40425531914893
    Report:                precision    recall  f1-score   support
    
               B       0.00      0.00      0.00        13
               L       0.73      0.79      0.76        85
               R       0.74      0.79      0.76        90
    
        accuracy                           0.73       188
       macro avg       0.49      0.53      0.51       188
    weighted avg       0.68      0.73      0.71       188
    
    Results Using Entropy:
    Predcited values:
    ['R' 'L' 'R' 'L' 'R' 'L' 'R' 'L' 'R' 'R' 'R' 'R' 'L' 'L' 'R' 'L' 'R' 'L'
     'L' 'R' 'L' 'R' 'L' 'L' 'R' 'L' 'R' 'L' 'R' 'L' 'R' 'L' 'R' 'L' 'L' 'L'
     'L' 'L' 'R' 'L' 'R' 'L' 'R' 'L' 'R' 'R' 'L' 'L' 'R' 'L' 'L' 'R' 'L' 'L'
     'R' 'L' 'R' 'R' 'L' 'R' 'R' 'R' 'L' 'L' 'R' 'L' 'L' 'R' 'L' 'L' 'L' 'R'
     'R' 'L' 'R' 'L' 'R' 'R' 'R' 'L' 'R' 'L' 'L' 'L' 'L' 'R' 'R' 'L' 'R' 'L'
     'R' 'R' 'L' 'L' 'L' 'R' 'R' 'L' 'L' 'L' 'R' 'L' 'L' 'R' 'R' 'R' 'R' 'R'
     'R' 'L' 'R' 'L' 'R' 'R' 'L' 'R' 'R' 'L' 'R' 'R' 'L' 'R' 'R' 'R' 'L' 'L'
     'L' 'L' 'L' 'R' 'R' 'R' 'R' 'L' 'R' 'R' 'R' 'L' 'L' 'R' 'L' 'R' 'L' 'R'
     'L' 'R' 'R' 'L' 'L' 'R' 'L' 'R' 'R' 'R' 'R' 'R' 'L' 'R' 'R' 'R' 'R' 'R'
     'R' 'L' 'R' 'L' 'R' 'R' 'L' 'R' 'L' 'R' 'L' 'R' 'L' 'L' 'L' 'L' 'L' 'R'
     'R' 'R' 'L' 'L' 'L' 'R' 'R' 'R']
    Confusion Matrix:  [[ 0  6  7]
     [ 0 63 22]
     [ 0 20 70]]
    Accuracy:  70.74468085106383
    Report:                precision    recall  f1-score   support
    
               B       0.00      0.00      0.00        13
               L       0.71      0.74      0.72        85
               R       0.71      0.78      0.74        90
    
        accuracy                           0.71       188
       macro avg       0.47      0.51      0.49       188
    weighted avg       0.66      0.71      0.68       188
    
    


```python

```
