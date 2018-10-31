#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

from scipy.fftpack import ifft2, fft2


# transformada de Fourier de la imagen del arbol
arbol = fft2(img.imread('arbol.png'))


# grafica de la transformada de Fourier del arbolito
fig = plt.figure()
ax = fig.gca()
#para ploterar los valores absolutos de los datos de la transformada
ax.imshow(np.abs(arbol))
#ejes de las graficas
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid()
ax.set_aspect('equal')
#guarda la imagen de la transformada
fig.savefig('DelgadoMaria_FT2D.pdf', type='pdf')


# In[ ]:




