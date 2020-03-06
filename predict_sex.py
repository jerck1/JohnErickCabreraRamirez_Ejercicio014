#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('(ls)')


# In[49]:


import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import f1_score


# In[16]:


data=glob.glob("./train/*.jpg")
y_vals=[]
for i in range(len(data)):
 #   if(i%2==0):
    if(int(data[i][8:-4])%2==0) :
        y_vals=np.append(y_vals,1)
    else:
        y_vals=np.append(y_vals,0)


# In[17]:


print(data[10][8:-4])


# In[61]:


files_train, files_test, y_train, y_test = train_test_split(data, y_vals, train_size=0.5)
print(files_train)
print(y_train)


# In[46]:


#files_train.sort()
images_train=[]
for f in files_train:
    i1=plt.imread(f)[:,:,0]
    i2=plt.imread(f)[:,:,1]
    i3=plt.imread(f)[:,:,2]
    d1=np.float_(i1.flatten())
    d2=np.float_(i2.flatten())
    d3=np.float_(i3.flatten())
    images_train.append((d1+d2+d3)/3)
#print(i[:,1])
print(np.shape(images_train))
#files_train.sort()
images_test=[]
for f in files_test:
    i1=plt.imread(f)[:,:,0]
    i2=plt.imread(f)[:,:,1]
    i3=plt.imread(f)[:,:,2]
    d1=np.float_(i1.flatten())
    d2=np.float_(i2.flatten())
    d3=np.float_(i3.flatten())
    images_test.append((d1+d2+d3)/3)
#print(i[:,1])
print(np.shape(images_test))


# In[24]:


n_test=len(files_test)
predict_test=np.int_(np.random.random(n_test)/0.5)
print(predict_test)
print(n_test)


# In[28]:


for f,p in zip(files_test,predict_test):
    print(f.split("/")[-1],p)


# In[ ]:


#cov = np.cov(images_train.T)
#images_train=np.array(images_train)
#cov = np.cov(images_train.T)
#valores, vectores = np.linalg.eig(cov)
#valores = np.real(valores)
#vectores = np.real(vectores)
#ii = np.argsort(-valores)
#valores = valores[ii]
#vectores = vectores[:,ii]
#test_trans = images_train.T @ vectores
#np.shape(images_train)
#np.shape(vectores)
#np.shape(test_trans)


# In[64]:


count = 0

for c in np.logspace(-4,10):    
    #Create a svm Classifier
    clf = SVC( C = c  , kernel='linear' ) # Linear Kernel

    #Train the model using the training sets
    clf.fit(images_train, y_train)

    # predigo los valores para test
    y_predict = SVC.predict(images_test)

    f1_array.append( f1_score(y_test, y_predict ) )

F1 = np.array(f1_array)
ii = np.argmax(f1_array)
C_max = c_array[ii]

svm = SVC( C=C_max, kernel='linear')
svm.fit(images_train, y_train)
y_pred = svm.predict(images_test)


# In[47]:


f1=[]
for c in np.logspace(-4,10):
    clf = SVC(C=c,kernel='linear',gamma='auto')
    clf.fit(images_test,y_test)
    predicted=clf.predict(images_test)
#    y_predict = clf.predict_proba(
#    f1_score(y_test,y_predict)
    print(predicted)


# In[ ]:




