
# coding: utf-8

# # Forming correlation matrices for all map shapes

# We'll get a list of all shapes in our sample

# In[1]:


#import required modules
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/raid5/homes/sshamsi/galaxy_zoo/GZ3D_spiral_analysis/')

from spiral_galaxy import SpiralGalaxy
import os


# This function will create a correlation matrix given an emission map shape.

# In[2]:


def form_corr_matrix(shape):
    r, c = shape
    
    if r != c:
        raise ValueError("r != c, the dimestions of emission map aren't equal!")
        
    corr_matrix = np.zeros((r**2, r**2))
    
    for i in range(r**2):
        for j in range(i, r**2):
            if i == j:
                corr_matrix[i, j] = 1
                continue
                
            y1 = i // r
            x1 = i % r
            
            y2 = j // r
            x2 = j % r
            
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            
            if dist < 6.4:
                rho = np.exp(-0.5 * (dist / 1.9)**2)
            else:
                rho = 0
                
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho
            
    return corr_matrix


# In[3]:


def append_files(dir_path):
    size_list = []
    
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".npy"):
                size = file.split('.')[0][-2:]
                size_list.append(size)
                
    return size_list


# In[4]:


sizes_done = append_files('/raid5/homes/sshamsi/galaxy_zoo/GZ3D_spiral_analysis/forming_cov_matrices/corr_matrices/')
available_gals = np.load('/raid5/homes/sshamsi/galaxy_zoo/GZ3D_spiral_analysis/available_v3_paths.npy')


# In[6]:


sizes_done


# In[7]:


available_gals


# In[ ]:


count = 0

for path in available_gals:
    count += 1
    print(count)
    
    gal = SpiralGalaxy(path)
    shape = gal.hamap.shape
    ifusize = shape[0]
    
    if str(ifusize) in sizes_done:
        print('Size already done.')
        continue
        
    corr_matrix = form_corr_matrix(shape)
    
    np.save('corr_matrices/corr_matrix' + str(ifusize), corr_matrix)
    print('New size corr saved!')


# In[7]:


type(sizes_done[0])

