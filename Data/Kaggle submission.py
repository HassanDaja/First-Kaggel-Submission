

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder



train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
del train["PassengerId"]
del test["PassengerId"]
train['Age'].fillna((train['Age'].mean()), inplace=True)




train["Sex_num"]=train.Sex.apply (lambda x:1 if x=="male" else 0)
train.head(10)









train.dropna(subset = ["Survived"], inplace=True)
train["Name_title"]=train.Name.apply (lambda x:x.split(",")[1].split(".")[0].strip())
train.head()




print(pd.pivot_table(train, index = 'Survived', columns = 'Name_title', values = 'Ticket' ,aggfunc ='count'))




most=['Miss','Lady','Mrs','Ms']
train["title_num"]=train.Name_title.apply (lambda x:1 if str(x) in most else 0)
train.head(10)




train["ticket_num"]=train.Ticket.apply (lambda x:1 if x.isdigit() else 0)
train.head()




print(test.isnull().sum())




test['Age'].fillna((test['Age'].mean()), inplace=True)




print(train.isnull().sum())




test['Fare'].fillna((train['Fare'].mean()), inplace=True)




train['Embarked'].fillna((train['Embarked'].mode()), inplace=True)




import seaborn as sns 
sns.barplot(train["Embarked"].value_counts().index,train["Embarked"].value_counts())





train.Embarked = train.Embarked.fillna('Q')




print(test.isnull().sum())




most=['Miss','Lady','Dr','Mrs']
test["Sex_num"]=test.Sex.apply (lambda x:1 if x=="male" else 0)
test["Name_title"]=test.Name.apply (lambda x:x.split(",")[1].split(".")[0].strip())
test["title_num"]=test.Name_title.apply (lambda x:1 if str(x) in most else 0)
test["ticket_num"]=test.Ticket.apply (lambda x:1 if x.isdigit() else 0)




train.head()




test.head(10)




test["num_cabins"]=test.Cabin.apply (lambda x:len(str(x).split(' ')))
train["num_cabins"]=train.Cabin.apply (lambda x:len(str(x).split(' ')))









del train["Ticket"]
del test["Ticket"]




test.head()




test.head()




target="Survived"
del train['Name']
del train['Cabin']
del test['Name']
del test['Cabin']




x=train.drop(target,1)
y=train[target]




objs=x.select_dtypes(include=["object"])
not_objs=x.select_dtypes(exclude=['object'])









la=LabelEncoder()
for x in range (objs.shape[1]):
    objs.iloc[:,x]=la.fit_transform( objs.iloc[:,x])




x=np.concatenate([objs,not_objs],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)









x_pre=test
objs=x_pre.select_dtypes(include=["object"])
not_objs=x_pre.select_dtypes(exclude=['object'])
la=LabelEncoder()
for x in range (objs.shape[1]):
    objs.iloc[:,x]=la.fit_transform(objs.iloc[:,x])
x_pre=np.concatenate([objs,not_objs],axis=1)
test.head()




Lpip=Pipeline([('myscaler',MinMaxScaler()),('mypca',PCA(n_components=3)),('logistic_classifier',LogisticRegression())])
Spip=Pipeline([('myscaler',MinMaxScaler()),('mypca',PCA(n_components=3)),('logistic_classifier',SVC())])
Rpip=Pipeline([('myscaler',MinMaxScaler()),('mypca',PCA(n_components=3)),('logistic_classifier',RandomForestClassifier())])
Kpip=Pipeline([('myscaler',MinMaxScaler()),('mypca',PCA(n_components=3)),('logistic_classifier',KNeighborsClassifier())])
Xpip=Pipeline([('myscaler',MinMaxScaler()),('mypca',PCA(n_components=3)),('logistic_classifier',XGBClassifier())])




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
    




model=testing(x_train,x_test,y_train,y_test)
model.fit(x_train,y_train)
prediction=model.predict(x_pre)




res=pd.read_csv("gender_submission.csv")




for x in range(len(prediction)):
    res['Survived'][x]=round(prediction[x])
    print(round(prediction[x]))
res.to_csv("result.csv", index=False)






