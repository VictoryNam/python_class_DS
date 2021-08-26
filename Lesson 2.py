#!/usr/bin/env python
# coding: utf-8

# ## Numpy
# 

# Task 1

# In[7]:


import numpy as np


# In[4]:


a = np.array([[1, 6],
            [2, 8],
            [3, 11],
            [3, 10],
            [1, 7]])
a.shape


# In[6]:


mean_a = (a.mean(axis = 0))


# In[16]:


print(f'The average of the columns: {mean_a}')


# Task 2

# In[11]:


a_centered = a - mean_a
a_centered 


# Task 3

# In[13]:


a1 = a_centered[:, 0].copy()
a2 = a_centered[:, 1].copy()
a_centered_sp = np.dot (a1, a2)


# In[18]:


print(f'result = {a_centered_sp/(len(a)-1)}')


# Task 4

# In[23]:


m = a.T
cov_a = np.cov(m)
print(f'covariance = {cov_a[0, 1]}')


# ## Pandas

# Task 1

# In[24]:


import pandas as pd


# In[26]:


authors = pd.DataFrame({'author_id': [1, 2, 3],
    'author_name': ['Turgenev', 'Chehov', 'Ostrovskiy'],
})
authors


# In[27]:


books = pd.DataFrame ({'author_id': [1, 1, 1, 2, 2, 3, 3], 
                      'book_title': ['Ottsy i deti', 'Rudin', 'Dvoriynskoe gnezdo', 'Tolstiy i tonkiy', 'Dama s sobachkoiy', 
                                    'Groza', 'Talanty i poklonniki'],
                      'price': [450, 300, 350, 500, 450, 370, 290]})
books


# Task 2

# In[28]:


authors_price = pd.merge(authors, books, on='author_id', how='inner')
authors_price


# Task 3

# In[39]:


top5 = authors_price.sort_values(by = ['price'], ascending = False).reset_index(drop = True)
top5.head(5)


# Task 4

# In[57]:


groupby_a = authors_price.groupby("author_name")


# In[70]:


authors_stat = (groupby_a.agg({"price": ['min', 'max', 'mean']})).rename({'min':'min_price', 
                                                                          'max':'max_price', 
                                                                          'mean':'mean_price'}, axis=1)
authors_stat


# Task 5

# In[82]:


cover_str = ['твердая', 'мягкая', 'мягкая', 'твердая', 'твердая', 'мягкая', 'мягкая']
authors_price['cover'] = cover_str
authors_price


# In[83]:


get_ipython().run_line_magic('pinfo', 'pd.pivot_table')


# In[85]:


import numpy as np


# In[94]:


book_info = pd.pivot_table(authors_price,
               index=['author_name'],
              columns=['cover'],
              values=['price'],
              aggfunc=[np.sum],
              fill_value=0)
book_info


# In[99]:


book_info.to_pickle("book_info.pkl")


# In[101]:


book_info2 = pd.read_pickle("book_info.pkl")
book_info2


# In[ ]:




