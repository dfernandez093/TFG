#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:32:05 2020

@author: Diego Fernánde Pérez

Este programa carga la regresión logísitica entrenada, y obtiene las precisiones sobre
los datos que se desee.
Es necesario que el tamaño de la dataset de prueba coincida con la que se usó 
para entrenar el modelo
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


#Cargamos el modelo entrenado
nombre = 'logreg_SGD_60x60_1000T.sav'
logreg_SGD = pickle.load(open(nombre, 'rb'))

#Cargamos los datos de test
datos=pd.read_pickle('spins_60x60_triangular_100_145.pkl')
fase=pd.read_pickle('fase_60x60_triangular_100_145.pkl')
mag=pd.read_pickle('mag_60x60_triangular_100_145.pkl')

#cargamos o creamos los puntos de temperatura
#T=pd.read_pickle('T_100.pkl')
ntemp=100
T = np.linspace(1,4.5, ntemp)

#Calculamos la precisión y probabilidades del modelo
precision=logreg_SGD.score(datos,fase)
prob=logreg_SGD.predict_proba(datos)


#representamos gráficamente la probabilidades y la magnetizción
fig, ax = plt.subplots()
ax.plot(T, prob[:,0], 'o', color="b",label='Prob. ser 1',markersize=3)
ax.plot(T, prob[:,1], '+', color="g",label='Prob. ser 0')
ax.plot(T, np.abs(mag), '*', color="r",label='magetización', markersize=3)
ax.axvspan(3,3.6,linewidth=4, color='grey',alpha=0.5)
major_ticks = np.arange(1, 5, 0.5)
ytick=['Desordenada','Ordenada']
y=np.array([0,1])
ax.set_xticks(major_ticks)
ax.set_yticks(y)
#ax.set_yticklabels(ytick,fontsize=6,verticalalignment='center')
ax.set_xlabel("Temperatura", fontsize=12)
ax.set_ylabel('Probabilidad',fontsize=12)
ax.legend()

plt.show()