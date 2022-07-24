#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plot
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[3]:


animals=pd.read_csv("E:/Placement courses/ML Projects/Horse Survival/horse.csv")


# In[4]:


target=animals["outcome"]


# In[5]:


target.unique()


# In[8]:


animals=animals.drop(['outcome'],axis=1)


# In[9]:


category_variables=["surgery","age","temp_of_extremities","peripheral_pulse","mucous_membrane",
                   "capillary_refill_time","pain","peristalsis","abdominal_distention","nasogastric_tube",
                   "nasogastric_reflux","rectal_exam_feces","abdomen",
                   "abdomo_appearance","surgical_lesion","cp_data"]

for category in category_variables:
    animals[category]=pd.get_dummies(animals[category])


# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

X,y=animals.values,target.values

label_encoder=LabelEncoder()
y=label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=1)


# In[11]:


from sklearn.tree import DecisionTreeClassifier
print(X_train.shape)


# In[14]:


from sklearn.impute import SimpleImputer
import numpy as np
imp=SimpleImputer(missing_values=np.nan, strategy="most_frequent")
X_train=imp.fit_transform(X_train)
X_test=imp.fit_transform(X_test)


# In[15]:


classifier=DecisionTreeClassifier()


# In[16]:


classifier.fit(X_train,y_train)


# In[17]:


y_predict=classifier.predict(X_test)


# In[18]:


from sklearn.metrics import accuracy_score


# In[19]:


accuracy=accuracy_score(y_predict,y_test)


# In[20]:


print(accuracy)


# In[21]:


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()


# In[22]:


classifier.fit(X_train,y_train)
y_predict=classifier.predict(X_test)
accuracy=accuracy_score(y_predict,y_test)
print(accuracy)


# In[ ]:




