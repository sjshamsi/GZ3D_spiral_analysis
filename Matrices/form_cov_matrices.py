#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, '/home/sshamsi_haverford_edu/galaxy_zoo/GZ3D_spiral_analysis/')


# In[2]:


import sys
sys.path.insert(0, '/home/sshamsi_haverford_edu/galaxy_zoo/Spiral_Reduce_Memory/')


# In[3]:


import spiral_resources
import sfr_methods


# In[4]:


import numpy as np
import os


# In[5]:


def append_files(dir_path):
    cov_list = []
    
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".npy"):
                filename = file.split('.')[0]
                cov_list.append(filename)
                
    return cov_list


# In[6]:


available_v3_paths = np.load('/home/sshamsi_haverford_edu/galaxy_zoo/GZ3D_spiral_analysis/available_v3_paths.npy')


# In[7]:


cov_done = append_files('/home/sshamsi_haverford_edu/galaxy_zoo/GZ3D_spiral_analysis/Matrices/cov_matrices/')


# In[8]:


available_v3_paths_working = np.load('available_v3_paths_working.npy')
available_v3_paths_notworking = np.load('available_v3_paths_notworking.npy')


# In[9]:


available_v3_paths_working


# In[ ]:


total_count = 0

for path in available_v3_paths:
    total_count += 1
    print('Total: ', total_count)
    
    filename = path.split('/')[-1].split('.')[0]
    
    if ((filename + 'ha') in cov_done) and ((filename + 'hb') in cov_done):
        if path not in available_v3_paths_working:
            np.append(available_v3_paths_working, path)
            
        print('Already done\n')
        continue
    
    try:
        galdict = spiral_resources.return_dict(path)
        galdf = spiral_resources.return_df(galdict)
        
        ha_cov = sfr_methods.cov_matrix_maker(galdict['map_shape'], galdf['$\sigma H_{\\alpha}$'])
        hb_cov = sfr_methods.cov_matrix_maker(galdict['map_shape'], galdf['$\sigma H_{\\beta}$'])
        
        np.save('cov_matrices/' + filename + 'ha', ha_cov)
        np.save('cov_matrices/' + filename + 'hb', hb_cov)
        
        if path not in available_v3_paths_working:
            np.append(available_v3_paths_working, path)
        
    except:
        print(filename, " didn't work, whoops.")
        
        if path not in available_v3_paths_notworking:
            np.append(available_v3_paths_notworking, path)
        

np.save('/home/sshamsi_haverford_edu/galaxy_zoo/GZ3D_spiral_analysis/Matrices/available_v3_paths_working', available_v3_paths_working)
np.save('/home/sshamsi_haverford_edu/galaxy_zoo/GZ3D_spiral_analysis/Matrices/available_v3_paths_notworking', available_v3_paths_notworking)


# In[ ]:




