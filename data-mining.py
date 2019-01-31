#!/usr/bin/env python
# coding: utf-8

# In[266]:


import pandas as pd
data = pd.read_excel('train.xlsx')


# In[267]:


X=data.iloc[:,0:67]
y=data.iloc[:,67]


# In[268]:


a=list(data.select_dtypes(include=['object']))
a.append('y')


# In[269]:


X_n = data.drop(a,axis =1).astype('float64')
print(X_n.shape)


# In[270]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
ax=y.value_counts().plot(kind='bar')
ax.set_ylabel('Observations')
ax.set_xlabel('class')
plt.show()


# In[232]:


list(data.select_dtypes(include=['object']))


# In[233]:


dummies = pd.get_dummies(data['x5'])
dummies_1 = pd.get_dummies(data['x13'])
dummies_2 = pd.get_dummies(data['x64'])
dummies_3 = pd.get_dummies(data['x65'])


# In[234]:


X_n.shape


# In[235]:


X=pd.concat([X_n,dummies[list(dummies)],dummies_1[list(dummies_1)],dummies_2[list(dummies_2)],
            dummies_3[list(dummies_3)]],axis=1).astype('float')


# In[236]:


# import numpy as np
# X.fillna(np.mean(X), inplace = True)
X.shape


# In[237]:


# data_1=data["y"]
# data = pd.concat([X,y],axis=1)
# from sklearn.utils import resample
# data_class1 = data[data_1==-1]
# data_class2 = data[data_1==1]
# data = resample(data_class2,n_samples=1891,replace=True, random_state = 123)
# data = pd.concat([data_class1,data])


# In[238]:


# X = data.drop('y',axis=1)
# y = data['y']


# In[239]:


from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
X = scale.fit(X).transform(X)


# In[240]:


X.shape


# In[241]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[92]:


from sklearn import svm
# model = svm.SVC(SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape='ovr', 
#                     degree=3, gamma='scale', kernel='rbf', max_iter=-1, probability=False, 
#                     random_state=None, shrinking=True, tol=0.001, verbose=False))
model = svm.SVC()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)


# In[93]:


y_pred.shape


# In[94]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[95]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))


# In[96]:


from sklearn.model_selection import GridSearchCV

model = svm.SVC()
model_grid = GridSearchCV(model, {'gamma':[0.00001,0.001,0.01,0.1,1,10,100], 'C':[0.01,0.1,1,10,100,1000]},return_train_score=True)
model_grid.fit(X_train, y_train)
model_best=model_grid.best_estimator_ 

print('Best parameters are:',model_grid.best_params_)


# In[97]:


import numpy as np
print("The Test Accuracy of the best RBF Kernel SVM is",np.round(model_best.score(X_test,y_test)*100,2),"%")


# In[98]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

y_pred = model_best.predict(X_test)
print(classification_report(y_test,y_pred))

cm = confusion_matrix(y_test, y_pred)
print("The Confusion Matrix looks like this: \n", cm)


# # Random forest

# In[242]:


y_train.value_counts()


# In[252]:


X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)


# In[264]:


data = pd.concat([X_train,y_train],axis=1).astype('float')
# data_1=data["y"]


# In[265]:


data


# In[261]:





from sklearn.utils import resample
data_class1 = data[data_1==-1]
data_class2 = data[data_1==1]


# In[262]:


data = resample(data_class2,n_samples=1516,replace=True, random_state = 123)
data = pd.concat([data_class1,data]).astype('float')


# In[222]:


X_train = data.drop('y',axis=1)
y_train = data['y']


# In[223]:


X_train.shape


# In[225]:


X_train


# In[ ]:



from sklearn.ensemble import RandomForestClassifier
modelnow=RandomForestClassifier()
modelnow.fit(X_train,y_train)
y_pred = modelnow.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print(classification_report(y_test,y_pred))

cm = confusion_matrix(y_test, y_pred)
print("The Confusion Matrix looks like this: \n", cm)


# In[ ]:





# In[ ]:




