#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import ifft
from scipy.interpolate import interp1d
archivo1 = 'signal.dat'
archivo2 = 'incompletos.dat'
delimitador = ','

#almacenamiento de los datos
senal1 = np.genfromtxt(archivo1, delimiter=delimitador, skip_header=0)
senal2 = np.genfromtxt(archivo2, delimiter=delimitador, skip_header=0)

#Grafica de la señal
plt.plot(archivo1)
plt.savefig("DelgadoMaria_signal.pdf")
# nuevos datos de las interpolaciones de los datos incompletos
nnuevo = 512

# define nuevo x
xnuevo = np.linspace(np.min(senal2[:,0]), np.max(senal2[:,0]), nnuevo)

# define nuevos Y
y3 = interp1d(senal2[:,0], senal2[:,1], 'quadratic')(xnuevo)
y4 = interp1d(senal2[:,0], senal2[:,1], 'cubic')(xnuevo)

# Agrupar xnuevo y los Y en senal3 y senal4
senal3 = np.column_stack((xnuevo,y3))
senal4 = np.column_stack((xnuevo,y4))

# Imprimir forma de los datos de las señales
print(np.shape(senal1), np.shape(senal2), np.shape(senal3), np.shape(senal4))

#Define la transformada de fourier
def mi_fourier(senal):
 # Extraer y de la senal
    y = senal[:,1]
#Cantidad de datos en y
    N = np.size(y,0) 
    
    # transfromada r que almacena la parte real y transformada i que almacena la imaginaria
    transformada_r = np.zeros(N) 
    transformada_i = np.zeros(N) 
    
# vector con los valores de N
    n = np.linspace(0, N-1, N) 
#Ciclo que calcula la parte real y imaginaria de la tranformada
    for k in range(N):
     
        transformada_r[k] = np.sum(y*np.cos(-2*np.pi*k*n/N))
        transformada_i[k] = np.sum(y*np.sin(-2*np.pi*k*n/N))
    
    # inicializa las frecuencias para almacenarlas
    frecuencias = np.zeros(N) 
#obtiene los valores que se encuentran en los datos
    x = senal[:,0]
# delta de la frecuencia
    D = 1.0/(2.0*(x[-1]-x[-2]))*(1/(N/2)) 
    
    
    
#Ciclo que genera las frecuencias a pasos de delta de las mismas. 
    for i in range(int(N/2)):
        frecuencias[i] = (i+1)*D
	#valores al reves 
    for i in range(-1,-int(N/2)-1, -1):
        frecuencias[i] = (i)*D
    #retorna los valores de la frecuencia y de la transformada real y imaginaria.
    return( frecuencias, transformada_r, transformada_i)


# Calcular fourier de la señal 1 llamando a la funcion mi fourier
f1, tf_r1, tf_i1 = mi_fourier(senal1) 
#calcula la amplitud de la primera señal
amplitud1 = (tf_r1**2 + tf_i1**2)**0.5 # calcular la amplitud

# Calcular fourier de la señal 2 llamando a la funcion mi fourier
f2, tf_r2, tf_i2 = mi_fourier(senal2) 
# calcular la amplitud de la segunda señal
amplitud2 = (tf_r2**2 + tf_i2**2)**0.5 


# Calcular fourier de la señal 3 llamando a la funcion mi fourier
f3, tf_r3, tf_i3 = mi_fourier(senal3) 
# calcular la amplitud de la tercera señal
amplitud3 = (tf_r3**2 + tf_i3**2)**0.5 

# Calcular fourier de la señal 4 llamando a la funcion mi fourier
f4, tf_r4, tf_i4 = mi_fourier(senal4) 
# calcular la amplitud de la cuarta señal
amplitud4 = (tf_r4**2 + tf_i4**2)**0.5 


#Imprime un mensaje por medio de un "print " para indicar que no se uso la herramienta fftfreq del paquete de Scipy
print("Mensaje para indicar que no se empleo fftfreq para calcular las frecuencias")

#Genera la grafica de las frecuencias y la guarda en formato PDF
fig = plt.figure()
ax = fig.gca()
ax.plot(f1, amplitud1)
ax.set_xlabel('Frecuencia')
ax.set_ylabel('Magnitud')
ax.grid()
ax.set_xlim([-2000,2000])
fig.savefig('DelgadoMaria_TF.pdf', type='pdf')

#Imprime el numero de picos de las frecuencias encontradas
print("Hay picos de frecuencias en 35, 180, 220 y 410")


# In[ ]:




