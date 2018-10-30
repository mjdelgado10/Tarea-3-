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

#Imprime el valor de la matriz de  covarianza

print(Covarianza, '\n\n', np.cov(variables.T))



#Autovalores y autovectores de la  matriz de covarianza
Valores_Vectores = np.linalg.eig(Covarianza) 

# Valores dados para autovalores y autoveectores de la matriz
#los autovalores corresponden a la columna 0 y los autovectores a la 1 
autovalores = Valores_Vectores[0]
autovectores = Valores_Vectores[1]


print("Los autovalores y autovectores son:")
for i in range(len(Covarianza)):
    print("\nAutovalor", i, autovalores[i], "\n   Autovector", i)
    for j in range(len(Covarianza)):
        print("  ", autovectores[j,i])


# In[ ]:




