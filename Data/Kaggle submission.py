#!/usr/bin/env python
# coding: utf-8

# In[730]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import warnings 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


# In[731]:


warnings.filterwarnings("ignore")


# In[ ]:





# In[732]:


train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
del train["PassengerId"]
del test["PassengerId"]
train['Age'].fillna((train['Age'].mean()), inplace=True)


# In[733]:


train["Sex_num"]=train.Sex.apply (lambda x:1 if x=="male" else 0)
train.head(10)


# In[ ]:





# In[734]:


train.dropna(subset = ["Survived"], inplace=True)
train["Name_title"]=train.Name.apply (lambda x:x.split(",")[1].split(".")[0].strip())
train.head()


# In[735]:


print(pd.pivot_table(train, index = 'Survived', columns = 'Name_title', values = 'Ticket' ,aggfunc ='count'))


# In[736]:


most=['Miss','Lady','Mrs','Ms']
train["title_num"]=train.Name_title.apply (lambda x:1 if str(x) in most else 0)
train.head(10)


# In[737]:


train["ticket_num"]=train.Ticket.apply (lambda x:1 if x.isdigit() else 0)
train.head()


# In[738]:


print(test.isnull().sum())


# In[739]:


test['Age'].fillna((test['Age'].mean()), inplace=True)


# In[740]:


print(train.isnull().sum())


# In[741]:


test['Fare'].fillna((train['Fare'].mean()), inplace=True)


# In[742]:


train['Embarked'].fillna((train['Embarked'].mode()), inplace=True)


# In[743]:


import seaborn as sns 
sns.barplot(train["Embarked"].value_counts().index,train["Embarked"].value_counts())


# In[744]:


train.Embarked = train.Embarked.fillna('Q')


# In[745]:


print(test.isnull().sum())


# In[746]:


most=['Miss','Lady','Dr','Mrs']
test["Sex_num"]=test.Sex.apply (lambda x:1 if x=="male" else 0)
test["Name_title"]=test.Name.apply (lambda x:x.split(",")[1].split(".")[0].strip())
test["title_num"]=test.Name_title.apply (lambda x:1 if str(x) in most else 0)
test["ticket_num"]=test.Ticket.apply (lambda x:1 if x.isdigit() else 0)


# In[747]:


train.head()


# In[748]:


test.head(10)


# In[749]:


test["num_cabins"]=test.Cabin.apply (lambda x:len(str(x).split(' ')))
train["num_cabins"]=train.Cabin.apply (lambda x:len(str(x).split(' ')))


# In[ ]:





# In[750]:


del train["Ticket"]
del test["Ticket"]


# In[751]:


test.head()


# In[752]:


test.head()


# In[753]:


target="Survived"
del train['Name']
del train['Cabin']
del test['Name']
del test['Cabin']


# In[754]:


x=train.drop(target,1)
y=train[target]


# In[755]:


objs=x.select_dtypes(include=["object"])
not_objs=x.select_dtypes(exclude=['object'])


# In[756]:


from sklearn.preprocessing import LabelEncoder


# In[757]:


la=LabelEncoder()
for x in range (objs.shape[1]):
    objs.iloc[:,x]=la.fit_transform( objs.iloc[:,x])


# In[758]:


x=np.concatenate([objs,not_objs],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)


# In[ ]:





# In[759]:


x_pre=test
objs=x_pre.select_dtypes(include=["object"])
not_objs=x_pre.select_dtypes(exclude=['object'])
la=LabelEncoder()
for x in range (objs.shape[1]):
    objs.iloc[:,x]=la.fit_transform(objs.iloc[:,x])
x_pre=np.concatenate([objs,not_objs],axis=1)
test.head()


# In[760]:


Lpip=Pipeline([('myscaler',MinMaxScaler()),('mypca',PCA(n_components=3)),('logistic_classifier',LogisticRegression())])
Spip=Pipeline([('myscaler',MinMaxScaler()),('mypca',PCA(n_components=3)),('logistic_classifier',SVC())])
Rpip=Pipeline([('myscaler',MinMaxScaler()),('mypca',PCA(n_components=3)),('logistic_classifier',RandomForestClassifier())])
Kpip=Pipeline([('myscaler',MinMaxScaler()),('mypca',PCA(n_components=3)),('logistic_classifier',KNeighborsClassifier())])
Xpip=Pipeline([('myscaler',MinMaxScaler()),('mypca',PCA(n_components=3)),('logistic_classifier',XGBClassifier())])


# In[761]:


def testing(x_train,x_test,y_train,y_test):
    models = {
              'LogReg': Lpip, 
              'RF': Rpip,
              'KNN': Kpip,
              'SVM': Spip, 
              'XGB': Xpip
    }
    name_score=[]
    for name,mod in models.items():
        mod.fit(x_train,y_train)
        name_score.append([model.score(x_test,y_test),name])
    
    for i in name_score:
        print(i)
    def Sort(sub_li): 

        # reverse = None (Sorts in Ascending order) 
        # key is set to sort using second element of 
        # sublist lambda has been used 
        sub_li.sort(key = lambda x: x[0],reverse=True) 
        return sub_li 

    return models[Sort(name_score)[0][1]]
    


# In[762]:


model=testing(x_train,x_test,y_train,y_test)
model.fit(x_train,y_train)
prediction=model.predict(x_pre)


# In[763]:


res=pd.read_csv("gender_submission.csv")


# In[764]:


for x in range(len(prediction)):
    res['Survived'][x]=round(prediction[x])
    print(round(prediction[x]))
res.to_csv("result.csv", index=False)


# In[ ]:




