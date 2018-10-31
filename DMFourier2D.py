#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

from scipy.fftpack import ifft2, fft2


# transformada de Fourier de la imagen del arbol
arbol = fft2(img.imread('arbol.png'))


# grafica de la transformada de Fourier del arbolito
fig = plt.figure()
ax = fig.gca()
#para ploterar los datos de la transformada
ax.imshow(np.abs(arbol))
#ejes de las graficas
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid()
ax.set_aspect('equal')
#guarda la imagen
fig.savefig('DelgadoMaria_FT2D.pdf', type='pdf')

#tamaño de la transformada
S = np.shape(arbol)

#Filtro para eliminar el ruido periodico


#Ciclo que recorre los datos de la imagen 
for i in range(S[0]):
#Amplitudes donde actua el ruido periodico
    for j in range(S[1]):
        # Ecuacion de linea recta de la banda de arriba
        if 13+1*i<j and j<17+1*i and i*j>100:
            arbol[i,j] = 0.001
            
        # Ecuacion de linea recta de la banda de la mitad
        if -2+1*i<j and j<2+1*i and i*j>100:
            arbol[i,j] = 0.001
            
        # Ecuacion de linea recta de la banda de abajo
        if -17+1*i<j and j<-13+1*i and i*j>100:
            arbol[i,j] = 0.001
#el filtrado final se obtiene como
#valor absoluto de los datos de la imagen
arbol_absoluto = np.abs(arbol)
#valor logaritmico de los valores absolutos de los datos
arbol_lognormal = np.log(arbol_absoluto)

#grafica que registra la transformada de fourier despues del filtrado
fig = plt.figure()
ax = fig.gca()
#grafica del filtrado
ax.imshow(arbol_lognormal)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid()
ax.set_aspect('equal')
#guarda las imagen de los datos filtrados
fig.savefig('DelgadoMaria_FT2D_filtrada', type='pdf')

#grafica de la transformada inversa de la imagen filtrada
fig = plt.figure()
ax = fig.gca()
#donde se realiza la transformada
ax.imshow(ifft2(arbol).real, cmap='gray')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid()
ax.set_aspect('equal')
fig.savefig('DelgadoMaria_filtrada', type='pdf')


# In[ ]:




