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

#grafica de la señal
fig = plt.figure()
ax = fig.gca()
ax.plot(senal1[:,0], senal1[:,1])
ax.set_xlabel('tiempo')
ax.set_ylabel('Magnitud')
ax.grid()
fig.savefig('DelgadoMaria_signal.pdf', type='pdf')

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

#Definicion de filtros que pasan frecuencias de  corte de 1000 o 500
def filtrar1000(f, real, imaginario):
    if np.abs(f)>1000: # si la frecuencia es mayor a 1000, devuelvo 0
        return 0
    else:
        return real + 1j*imaginario # si no devuelvo el mismo valor

def filtrar500(f, real, imaginario):
    if np.abs(f)>500: # si la frecuencia es mayor a 1000, devuelvo 0
        return 0
    else:
        return real + 1j*imaginario # si no devuelvo el mismo valor
#Se aplica el filtro solo a tres amplitudes    
# Aplicar el filtro a cada elemento de la amplitud y se convierte en array 
amplitud1_filtrada1000 = [filtrar1000(f1[i],tf_r1[i],tf_i1[i]) for i in range(len(f1))]
amplitud1_filtrada1000 = np.array(amplitud1_filtrada1000)

amplitud1_filtrada500 = [filtrar500(f1[i],tf_r1[i],tf_i1[i]) for i in range(len(f1))]
amplitud1_filtrada500 = np.array(amplitud1_filtrada500)


# Aplicar el filtro a cada elemento de la amplitud y convertirlo en array
amplitud3_filtrada1000 = [filtrar1000(f3[i],tf_r3[i],tf_i3[i]) for i in range(len(f3))]
amplitud3_filtrada1000 = np.array(amplitud3_filtrada1000)

amplitud3_filtrada500 = [filtrar500(f3[i],tf_r3[i],tf_i3[i]) for i in range(len(f3))]
amplitud3_filtrada500 = np.array(amplitud3_filtrada500)


# Aplicar el filtro a cada elemento de la amplitud y convertirlo en array
amplitud4_filtrada1000 = [filtrar1000(f4[i],tf_r4[i],tf_i4[i]) for i in range(len(f4))]
amplitud4_filtrada1000 = np.array(amplitud4_filtrada1000)

amplitud4_filtrada500 = [filtrar500(f4[i],tf_r4[i],tf_i4[i]) for i in range(len(f4))]
amplitud4_filtrada500 = np.array(amplitud4_filtrada500)


# trasnformada inversa de las 3 señales
senal1_filtrada1000 = ifft(amplitud1_filtrada1000)
senal1_filtrada500 = ifft(amplitud1_filtrada500)
senal1_filtradax = senal1[:,0]

senal3_filtrada1000 = ifft(amplitud3_filtrada1000)
senal3_filtrada500 = ifft(amplitud3_filtrada500)
senal3_filtradax = senal3[:,0]

senal4_filtrada1000 = ifft(amplitud4_filtrada1000)
senal4_filtrada500 = ifft(amplitud4_filtrada500)
senal4_filtradax = senal4[:,0]


# grafica para la primera senal filtrada que es la senal 1
fig = plt.figure()
ax = fig.gca()
ax.plot(senal1_filtradax, senal1_filtrada1000.real)
ax.set_xlabel('tiempo')
ax.set_ylabel('Magnitud')
ax.grid()
ax.set_xlim([0,0.03])
fig.savefig('DelgadoMaria_filtrada.pdf', type='pdf')

#Mensaje donde se indica que no se puede hacer la transformada para lncompletos.dat
print("En los datos incompletos no tiene sentido hacer la transformada de Fourier porque el periodo de muestreo no es constante y en consecuencia no se pueden identificar los armonicos")


#Interpolacion de los datos 
# grafica con los datos interpolados antes de las tres transformadas de fourier
fig = plt.figure()

fig.add_subplot(311)
ax = fig.gca()
ax.plot(f1, amplitud1)
ax.set_ylabel('Magnitud')
ax.legend(['senal'])
ax.grid()

#interpolacion cuadratica
fig.add_subplot(312)
ax = fig.gca()
ax.plot(f3, amplitud3)
ax.set_ylabel('Magnitud')
ax.legend(['cuadrado'])
ax.grid()

#Interpolacion cubica
fig.add_subplot(313)
ax = fig.gca()
ax.plot(f4, amplitud4)
ax.set_ylabel('Magnitud')
ax.legend(['cubico'])
ax.grid()

fig.savefig('DelgadoMaria_TF_interpola.pdf', type='pdf')

#Diferencias entre las transformadas de fourier de la señal  normal y de sus interpolaciones 
print("El espectro de frecuencias de la interpolacion cubica es similar al espectro de la senal original")
print("solo que adiciona ruido de alta frecuencia.")
print("El espectro de frecuencias de la interpolacion cuadrada tambien es similar al espectro de la senal original")
print("pero ademas de adicionar ruido de alta frecuencia, tambien adiciona ruido de baja frecuencia y alta magnitud")


# grafica de las señales filtradas
fig = plt.figure()

fig.add_subplot(211)
ax = fig.gca()
ax.plot(senal1_filtradax, senal1_filtrada1000.real)
ax.plot(senal3_filtradax, senal3_filtrada1000.real)
ax.plot(senal4_filtradax, senal4_filtrada1000.real)
ax.set_ylabel('Magnitud')
ax.legend(['1000-senal', '1000-cuadrado', '1000-cubico'])
ax.grid()

fig.add_subplot(212)
ax = fig.gca()
ax.plot(senal1_filtradax, senal1_filtrada500.real)
ax.plot(senal3_filtradax, senal3_filtrada500.real)
ax.plot(senal4_filtradax, senal4_filtrada500.real)
ax.set_ylabel('Magnitud')
ax.legend(['500-senal', '500-cuadrado', '500-cubico'])
ax.grid()

fig.set_size_inches(10,5)

fig.savefig('DelgadoMaria_2Filtros.pdf', type='pdf')


# In[ ]:




