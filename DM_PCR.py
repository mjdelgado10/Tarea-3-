#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Regresion PCR
import numpy as np
import matplotlib.pyplot as plt


columnas = range(2,32) # Columnas de los datos para cargar


#Almacenar los datos del archivo WDBC.dat
MalignoBenigno = np.genfromtxt('WDBC.dat', dtype='str', usecols=1, delimiter=',')
#Separar variables de los datos, maligno y benigno.
variables = np.genfromtxt('WDBC.dat', usecols=columnas, delimiter=',')

#tama�o de las variables 
n,m = np.shape(variables)

#Inicializacion para empezar la covarianza
Covarianza = [] 


# In[ ]:




