#!/usr/bin/env python
# coding: utf-8

# In[3]:


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
#Inicio del ciclo que recorrera todos los datos
for i in range(m):
    fila_matriz = [] # lista vacia
#Ciclo que genera el promedio de i y j de la variable m
    for j in range(m):
        promedio_i = np.mean(variables[:,i])
        promedio_j = np.mean(variables[:,j])
#genera la covarianza de las variables segun la formula. 
        Covar = np.sum( (variables[:,i]-promedio_i) * (variables[:,j]-promedio_j) )/(n-1)
# Calcula cada fila de la matriz y la guarda
        fila_matriz.append(Covar)
    Covarianza.append(fila_matriz)
#Una vez se llena la lista, se crea el array
Covarianza = np.array(Covarianza)

#Imprime el valor de la covarianza

print(Covarianza, '\n\n', np.cov(variables.T))




# In[ ]:




