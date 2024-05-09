# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree classification in dataset.
4. calculate Accuracy,data prediction.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: muni tejeshwar
RegisterNumber: 212223040102
*/

import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company",
          "Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
![image](https://github.com/SanjithaBolisetti/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393633/e0a7971b-4af6-4e54-9aeb-a44a03edfdcb)

![image](https://github.com/SanjithaBolisetti/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393633/e5786d97-2d64-4e2f-a91b-6af899ab22f6)

![image](https://github.com/SanjithaBolisetti/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393633/4410659a-5932-475b-a683-88b62ab650f7)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
