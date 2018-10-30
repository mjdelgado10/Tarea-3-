#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
archivo1 = 'signal.dat'
archivo2 = 'incompletos.dat'
delimitador = ','

#almacenamiento de los datos
senal1 = np.genfromtxt(archivo1, delimiter=delimitador, skip_header=0)
senal2 = np.genfromtxt(archivo2, delimiter=delimitador, skip_header=0)

#Grafica de la señal
plt.plot(archivo1)
plt.savefig("DelgadoMaria_signal.pdf")


# In[ ]:




