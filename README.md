# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
## STEP 1
Import the required libraries.
## STEP 2
Read the data frame using pandas.
## STEP 3
Get the information regarding the null values present in the dataframe.
## STEP 4
Split the data into training and testing sets.
## STEP 5
Convert the text data into a numerical representation using CountVectorizer.
## STEP 6
Use a Support Vector Machine (SVM) to train a model on the training data and make predictions on the testing data.
## STEP 7
Finally, evaluate the accuracy of the model.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: JEEVITHA E
RegisterNumber:  212222230054
*/
```
```
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
``

## Output:


data.head()



![image](https://github.com/sreenithi23/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147017600/cdef2a29-9188-417c-923f-49266608b418)


data.info()


![image](https://github.com/sreenithi23/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147017600/0e8e2293-7372-4baa-aec5-146d404ee913)

Data.isnull().sum()



![image](https://github.com/sreenithi23/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147017600/35a35e67-79fa-4c85-ab44-cf8d7d6d4fdf)

y_pred


![image](https://github.com/sreenithi23/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147017600/39fdaffc-43b6-4995-b98a-a1e13cc381a1)

accuracy



![image](https://github.com/sreenithi23/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147017600/4ec6c7cb-35e7-4dbe-8516-c955fbc98bb2)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
