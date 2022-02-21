import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
data=pd.read_csv("student-por.csv",sep=";")
target='G3'
x=data.drop(target,1)
y=data[target]
objs=x.select_dtypes(include=["object"])
not_objs=x.select_dtypes(exclude=['object'])
la=LabelEncoder()
for x in range (objs.shape[1]):
    objs.iloc[:,x]=la.fit_transform( objs.iloc[:,x])
x=np.concatenate([objs,not_objs],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.7)
model=LinearRegression()
model=model.fit(x_train,y_train)
prediction=model.predict(x_test)
for pre,real in zip(prediction,y_test):
    print(f"Predicted:{int(pre)} & Real:{real}")


