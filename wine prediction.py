#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


wine = pd.read_csv('C:\wine\winequality-red.csv')


# In[3]:


wine.head()


# In[4]:


wine.info()


# In[5]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = wine)


# In[6]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'volatile acidity', data = wine)


# In[7]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'citric acid', data = wine)


# In[8]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'residual sugar', data = wine)


# In[9]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'chlorides', data = wine)


# In[10]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'alcohol', data = wine)


# In[11]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'sulphates', data = wine)


# In[12]:


bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)


# In[13]:


label_quality = LabelEncoder()


# In[15]:


wine['quality'] = label_quality.fit_transform(wine['quality'])


# In[17]:


wine['quality'].value_counts()


# In[18]:


sns.countplot(wine['quality'])


# In[19]:


X = wine.drop('quality', axis = 1)
y = wine['quality']


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[21]:


sc = StandardScaler()


# In[22]:


X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[23]:


rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)


# In[24]:


print(classification_report(y_test, pred_rfc))


# In[25]:


print(confusion_matrix(y_test, pred_rfc))


# In[26]:


sgd = SGDClassifier(penalty=None)
sgd.fit(X_train, y_train)
pred_sgd = sgd.predict(X_test)


# In[27]:


print(classification_report(y_test, pred_sgd))


# In[28]:


print(confusion_matrix(y_test, pred_sgd))


# In[29]:


svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)


# In[30]:


print(classification_report(y_test, pred_svc))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




