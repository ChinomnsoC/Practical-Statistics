#!/usr/bin/env python
# coding: utf-8

# ### Interpreting Coefficients
# 
# It is important that not only can you fit complex linear models, but that you then know which variables you can interpret. 
# 
# In this notebook, I fitted two different models and interpreted their results
# 

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm;

df = pd.read_csv('./house_prices.csv')
df.head()


# 
# ### Model 1
# 
# `1.` For the first model, I will fit a model to predict `price` using `neighborhood`, `style`, and the `area` of the home.  I will build your dummy variables, use and intercept and drop one of the columns to fit the linear model. The baselines include neighborhood 'C' and home style **lodge**.

# In[3]:


#adding dummy variables for the style and neighborhood of house
df[['lodge', 'ranch', 'victorian']] = pd.get_dummies(df['style'])
df[['A', 'B', 'C']] = pd.get_dummies(df['neighborhood'])
df.head()


# In[4]:


#fiting a model to predict price using neighborhood, style, 
#and the area of the home.

#adding intercept
df['intercept'] = 1
#creating the linear model
lm = sm.OLS(df['price'], df[['intercept', 'A', 'B', 'ranch', 'victorian']])
#the baselines are neighborhood C and home style lodge

results = lm.fit()
results.summary()


# ### Model 2
# 
# `2.` In this second model for predicting price, I will use `area` and `area squared` to predict price.  Also I will use the `style` of the home, but not `neighborhood` this time. I will use dummy variables, and add an intercept to the model. 

# In[5]:


#fitting area squared column
df['area squared'] = df['area']*df['area']


# In[6]:


#fitting model 2 to predict price using area, area squared, and style
lm = sm.OLS(df['price'], df[['intercept', 'ranch', 'victorian', 'area squared', 'area']])
#lodge remains baseline
results = lm.fit()
results.summary()


# In[8]:


# Trying one more model, with neighborhoods also, responding to
#   prompt in question 3
lm = sm.OLS(df['price'], df[['intercept', 'A', 'B', 'ranch', 'victorian', 'area squared', 'area']])
results = lm.fit()
results.summary()


# In[ ]:




