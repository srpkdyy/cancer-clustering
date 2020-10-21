
# coding: utf-8

# In[14]:

import argparse
import numpy as np
import pandas as pd
import sklearn
import pandas
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import torch
import matplotlib
#get_ipython().run_line_magic('matplotlib', 'inline')
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--feature', type=str, default='latest')
args = parser.parse_args()

# In[2]:


data = np.load('result/features/' + args.feature, allow_pickle=True)
print('data.shape', data.shape)

# In[3]:


X, y = data[:, 0], data[:, 1]


# In[4]:


z = np.array([*X])
print('z.shape', z.shape)


# In[5]:


y_color = pd.DataFrame({'label':np.array(y, dtype=np.int)})


# In[6]:


color_table = ['#FF0000', '#0000FF']
type_table = ['a', 'h']
y_color['color'] = y_color['label'].apply(lambda x: color_table[x])
y_color['type'] = y_color['label'].apply(lambda x: type_table[x])


# In[7]:


tsne_z = z#TSNE(n_components=2).fit_transform(z)
print('tsne_z.shape', tsne_z.shape)


# In[8]:


plt.figure(figsize=(10, 10))
plt.scatter(tsne_z[:, 0], tsne_z[:, 1], color=y_color['color'])
plt.title('t-SNE')
plt.xlabel('The first score')
plt.ylabel('The second score')
plt.show()

# In[9]:


kmeans = KMeans(n_clusters=2).fit(z)
pred = kmeans.labels_
print(pred)


# In[10]:


#Acc 58.4 -> 59.4
count = [{'a': 0, 'h': 0} for _ in range(2)]
for i, pred_label in enumerate(pred):
    count[pred_label][y_color['type'][i]] += 1
print(count)
