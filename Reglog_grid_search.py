#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 19:53:03 2020

@author: Diego Fernández Pérez

Este programa sirve para entrenar el modelo de regresión lineal para
distintos tamaños de redes, hacer un grid search del hiperparametro
 y dibujar las probababilidades
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn import linear_model

#parametros 
T = np.linspace(0.25,4.50, 1000)
#tamaños de las redes que queramos comparar
N = np.array((40,50,60))
num_classes = 2
#ratio entre la data ser de prueba y entrenamiento
train_to_test_ratio = 0.8
#valores de lambda que queramos probar
lmbdas=np.logspace(-5,5,11)
#guardamos las precisiones para poder dibujar y comparar
train_accuracy_SGD=np.zeros((3,lmbdas.shape[0]))
test_accuracy_SGD=np.zeros((3,lmbdas.shape[0]))
for i,n in enumerate(N):
    data = pd.read_pickle('spins_%1.fx%1.f.pkl'%(n,n))
    fase= pd.read_pickle('fase_%1.fx%1.f.pkl'%(n,n))
    #solo es valido para los datos creado con el T especificado de 1000 puntos
    #entre 0.25 y 4.50
    X_orden= data[:473,:]
    Y_orden = fase[:473,0]
    X_desorden =  data[473:,:]
    Y_desorden = fase[473:,0]
    del data
    X=np.concatenate((X_orden,X_desorden))
    Y=np.concatenate((Y_orden,Y_desorden))
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=train_to_test_ratio,test_size=1-train_to_test_ratio)
    #hacemos el grid search
    for j, lmbda in enumerate(lmbdas):
    
        logreg_SGD = linear_model.SGDClassifier(loss='log', penalty='l2',alpha=lmbda, max_iter=100,tol=1E-5, shuffle=True,
                                           random_state=1, learning_rate='optimal')
    
        logreg_SGD.fit(X_train,Y_train)
    
    # guardamos la precisión
        train_accuracy_SGD[i,j]=logreg_SGD.score(X_train,Y_train)
        test_accuracy_SGD[i,j]=logreg_SGD.score(X_test,Y_test)
    
    
# dibujamos la precisión en función de los valores de lambda

plt.semilogx(lmbdas,train_accuracy_SGD[0,:],'*-',color='orange',label='40x40 train')
plt.semilogx(lmbdas,test_accuracy_SGD[0,:],'*-r',label='40x40 test')

plt.semilogx(lmbdas,train_accuracy_SGD[1,:],'o-y',label='50x50 train')
plt.semilogx(lmbdas,test_accuracy_SGD[1,:],'o-g',label='50x50 test')

plt.semilogx(lmbdas,train_accuracy_SGD[2,:],'+-r',label='60x60 train')
plt.semilogx(lmbdas,test_accuracy_SGD[2,:],'+-',color='purple',label='60x60 test')



plt.xlabel('$\\lambda$')
plt.ylabel('$\\mathrm{Precisión}$')

plt.grid()
plt.legend()


plt.show()


###############################################################################
#nos quedamos con la mejor red, la entrenamos y guardamos
logreg_SGD = linear_model.SGDClassifier(loss='log', penalty='l2',alpha=0.01, max_iter=100,tol=1E-5, shuffle=True,
                                           random_state=1, learning_rate='optimal')
    
logreg_SGD.fit(X_train,Y_train)

# guardamos la mejor red entrenada
nombre = 'logreg_SGD_60x60_1000T.sav'
pickle.dump(logreg_SGD, open(nombre, 'wb'))

#probamos la red y ploteamos
data_test = pd.read_pickle('spins_60x60_triangular.pkl')
fase_test= pd.read_pickle('fase_60x60_triangular.pkl')
mag=pd.read_pickle('mag_60x60_triangular_100.pkl')
logreg_SGD.score(data_test,fase_test)
temp=pd.read_pickle('T_100.pkl')
y_cor=logreg_SGD.predict(data_test)

plt.plot(T, y_cor, '+', color="b",label='y pred')
plt.plot(T, abs(np.array(mag)), '*', color="purple",label='Magentización')
plt.xlabel("Temperatura", fontsize=12)
plt.ylabel('Clasificación / Magnetización ', fontsize=12)
plt.grid()
plt.legend()
plt.savefig('magnetization_MC.png')
