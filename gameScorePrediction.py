#!/usr/bin/env python
# coding: utf-8

# In[450]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
get_ipython().run_line_magic('matplotlib', 'inline')


# In[453]:


df = pd.read_csv("data/gamedata.csv") 


# In[454]:


headers = ['Date','Level 1','Level 2','Level 3','Level 4','Level 5']


# In[457]:


df = pd.read_csv('data/gamedata.csv',names=headers)


# In[458]:


x = df['Date']
y1 = df['Level 1']
y2 = df['Level 2']
y3 = df['Level 3']
y4 = df['Level 4']
y5 = df['Level 5']

df = pd.read_csv("data/gamedata.csv") 

X = df.iloc[:, 0:1].values
y11 = df.iloc[:, 1].values
y12 = df.iloc[:, 2].values
y13 = df.iloc[:, 3].values
y14 = df.iloc[:, 4].values
y15 = df.iloc[:, 5].values
plt.rcParams["figure.figsize"] = (20,10)


# In[459]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y11, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
lin_reg1 = LinearRegression()
lin_reg1.fit(X, y11)


# In[460]:


ypoint1 = np.array([lin_reg1.predict([[20210609]])])
xpoint1 = np.array([20210609])
print(ypoint1)#predicted value


# In[461]:


plt.plot(xpoint1, ypoint1, '_',color = 'r',ms = 2500)
plt.plot(xpoint1, ypoint1, '|',color = 'r',ms = 2000)
plt.plot(x,y1)
plt.show()


# In[462]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y12, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
lin_reg2 = LinearRegression()
lin_reg2.fit(X, y12)


# In[463]:


ypoint2 = np.array([lin_reg2.predict([[20210609]])])
xpoint2 = np.array([20210609])
print(ypoint2)#predicted value


# In[464]:


plt.plot(xpoint2, ypoint2, '_',color = 'r',ms = 2500)
plt.plot(xpoint2, ypoint2, '|',color = 'r',ms = 2000)
plt.plot(x,y2)
plt.show()


# In[465]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y13, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
lin_reg3 = LinearRegression()
lin_reg3.fit(X, y13)


# In[466]:


ypoint3 = np.array([lin_reg3.predict([[20210609]])])
xpoint3 = np.array([20210609])
print(ypoint3)#predicted value


# In[467]:


plt.plot(xpoint3, ypoint3, '_',color = 'r',ms = 2500)
plt.plot(xpoint3, ypoint3, '|',color = 'r',ms = 2000)
plt.plot(x,y3)
plt.show()


# In[468]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y14, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
lin_reg4 = LinearRegression()
lin_reg4.fit(X, y14)


# In[469]:


ypoint4 = np.array([lin_reg4.predict([[20210609]])])
xpoint4 = np.array([20210609])
print(ypoint4)#predicted value


# In[470]:


plt.plot(xpoint4, ypoint4, '_',color = 'r',ms = 2500)
plt.plot(xpoint4, ypoint4, '|',color = 'r',ms = 2000)
plt.plot(x,y4)
plt.show()


# In[471]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y15, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
lin_reg5 = LinearRegression()
lin_reg5.fit(X, y15)


# In[472]:


ypoint5 = np.array([lin_reg5.predict([[20210609]])])
xpoint5 = np.array([20210609])
print(ypoint5)#predicted value


# In[473]:


plt.plot(xpoint5, ypoint5, '_',color = 'r',ms = 2500)
plt.plot(xpoint5, ypoint5, '|',color = 'r',ms = 2000)
plt.plot(x,y5)
plt.show()

