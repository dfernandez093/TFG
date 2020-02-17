#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:41:38 2020

@author: Diego Fernández

Simulacion y representación de configuraciones de red del modelo de Ising
"""

import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt

class Ising():
    ''' Simulación de configs. para el modelo de Ising '''    
    ## Creamos las configuraciones
    @njit(parallel=True) # permite el cálculo en paralelo
    def paso_metropolis(self,config,N,T):
        for _ in prange(N * N):
            #selección de un punto arbritrario de la red
            x, y = np.random.randint(0, N), np.random.randint(0, N)
            #Se tiene en cuenta las condiciones periodicas %
            deltaE = 2 * config[x, y] * (config[(x + 1) % N, y] + config[x, (y + 1) % N] + # Periodic boundary conditions
                               config[(x - 1) % N, y] + config[x, (y - 1) % N])
            if deltaE < 0:
                config[x, y] *= -1
            elif np.random.rand() < np.exp(-deltaE / T):
                config[x, y] *= -1
        return(config)
    def simulate(self):   
        ''' Simula los paso del algoritmo metropolis en la red'''
        N, temp     = 40, .5     # Valores iniciales de red y temperatura
        config = 2*np.random.randint(2, size=(N,N))-1
        f = plt.figure(figsize=(15, 15), dpi=80);    
        self.configPlot(f, config, 0, N, 1);
        
        msrmnt = 1001
        for i in range(msrmnt):
            self.paso_metropolis(config, N, temp)
            if i == 1:       self.configPlot(f, config, i, N, 2);
            if i == 4:       self.configPlot(f, config, i, N, 3);
            if i == 32:      self.configPlot(f, config, i, N, 4);
            if i == 100:     self.configPlot(f, config, i, N, 5);
            if i == 1000:    self.configPlot(f, config, i, N, 6);
                 
                    
    def configPlot(self, f, config, i, N, n_):
        ''' Representación de las configuraciones'''
        X, Y = np.meshgrid(range(N), range(N))
        sp =  f.add_subplot(3, 3, n_ )  
        plt.setp(sp.get_yticklabels(), visible=False)
        plt.setp(sp.get_xticklabels(), visible=False)      
        plt.pcolormesh(X, Y, config, cmap=plt.cm.RdBu);
        plt.title('Time=%d'%i); plt.axis('tight')    
    plt.show()
