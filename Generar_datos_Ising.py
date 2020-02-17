#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 22:02:45 2020

@author: Diego Fernández Pérez

Este código permite generar y guardar configuraciones de espines del modelo de Ising
siguiendo el algortimo de metropolis.
Es posible generar datos para dos tipos de interacciones, Primeros vecinos o hexagonales

"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
from numba import njit, prange
# devuelve una configuración de unos 
def estadoinicial(N):
    '''genera un configuración ordenada de spines'''
    state=np.ones((N,N))
    return(state)

# implementaccion del algoritmo metropolis
@njit(parallel=True) # permite el cálculo en paralelo
def paso_metropolis(config, T,inter='vecinos'):
    for _ in prange(N * N):
        #selección de un punto arbritrario de la red
        x, y = np.random.randint(0, N), np.random.randint(0, N)
        #Interacción hexagonal
        #Se tiene en cuenta las condiciones periodicas %
        if inter == 'trinagular':
            deltaE = 2 * config[x, y] * (config[(x+1)%N,x] + config[(x-1)%N,(y+1)%N] + config[(x-1)%N,y] + 
                               config[(x-1)%N,(y-1)%N]+config[(x+1)%N,(y+1)%N]+config[(x+1)%N,(y-1)%N])
        elif inter == 'vecinos':
            deltaE = 2 * config[x, y] * (config[(x + 1) % N, y] + config[x, (y + 1) % N] + # Periodic boundary conditions
                       config[(x - 1) % N, y] + config[x, (y - 1) % N])
        else:
            print('Interacción desconocida')
        if deltaE < 0:
            config[x, y] *= -1
        elif np.random.rand() < np.exp(-deltaE / T):
            config[x, y] *= -1


def crear_config(N,Tc,Mcsteps,EQsteps,ntemp,T):
    magnetizacion = []
    # inicializamos las variables
    spins = np.zeros((0, N * N))
    fase= np.zeros((ntemp,1))
    inicial=estadoinicial(N)
    for index, temp in enumerate(T):
        tmp = []
        config = inicial

        # alcanzamos el equilibrio en las configuraciones
        for _ in range(EQsteps):
            paso_metropolis(config, temp)

        
        for mc in range(MCsteps):
            #guardamos parte de los datos para calcular la magnetización
            if mc % 200 == 0:
                tmp.append(np.sum(config))
            paso_metropolis(config, temp)
        spins = np.vstack((spins, config.ravel()))

        # etiquetamos la configuración en base a la temperatura
        #1 ordenado ; 0 desordenado
        if temp < Tc:
            fase[index,0] = 1
        else:
            fase[index,0] =0
            
        magnetizacion.append(np.mean(tmp) / (N * N))
        print('{} out of {} temperature steps'.format(index, len(T)))
    return(spins,fase,magnetizacion)



# Para guardar los datos comprimidos en pickle
def save(object, nombre):
    with open(nombre + '.pkl', 'wb') as f:
        pickle.dump(object, f)



# lista de parámetros para generar los datos       
N = 60
Tc = 2.269
#pasos del algortimo metrópolis
MCsteps = 10000
#pasos hasta alcanzar el equilibrio
EQsteps = 2500
#número de puntos de tempertatura
ntemp = 100
T = np.linspace(1,4.5, ntemp)
spins,fase,mag = crear_config(N,Tc,MCsteps,EQsteps,ntemp,T)
#Guardamos los espines de forma binaria para mejorar memoria y velocidad
save(0.5 * (spins + 1), 'spins_%1.fx%1.f_triangular_%1.f_145'%(N,N,ntemp))
save(fase, 'fase_%1.fx%1.f_triangular_%1.f_145'%(N,N,ntemp))
save(mag,'mag_%1.fx%1.f_triangular_%1.f_145'%(N,N,ntemp))
#Comprovamos la valided de los datos representando la magnetización 
plt.plot(T, abs(np.array(mag)), 'o', color="r")
plt.xlabel("Temperatura", fontsize=12)
plt.ylabel("Magnetización ", fontsize=12)
plt.grid()