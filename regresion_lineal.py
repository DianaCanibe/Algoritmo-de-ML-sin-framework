#Regresión Lineal Simple 
#Diana Cañibe Valle   A01749422
'''Implementación del algoritmo de regresión lineal simple para 2 variables numéricas
sin el uso de un framework'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Modelo Regresión Lineal 
def regresion(m,x,b):
    return m*x+b

#Error cuadrático medio 
def error_medio(y,y_pred):
    error = np.sum((y-y_pred)**2)/len(y)
    return error

#Gradiente descendiente
def gradiente(m_pred,b_pred,alpha,x,y):
    N = float(len(x))
    y_pred= m_pred*x+b_pred
    #Cálculo del gradientes
    dm = -(2/N)*np.sum(x*(y-y_pred))
    db = -(2/N)*np.sum(y-y_pred)
    #Actualización de pesos
    m = m_pred - alpha*dm
    b = b_pred - alpha*db
    return m, b

def train_test(dataframe,part_train):
    part_train = dataframe.sample(frac = part_train)
    part_test = dataframe.drop(part_train.index)
    return part_train,part_test
    
#Set de datos de prueba
df= pd.read_csv('blood_pressure.csv')
#Partición del set de datos
train,test = train_test(df,0.8)

x_train = train['Age'].values
y_train = train['Systolic blood pressure'].values

x_test = test['Age'].values
y_test = test['Systolic blood pressure'].values

#Variables de inicio
'''Definicion aleatoria inicial para que el modelo aprenda'''
m = np.random.randn(1)[0]  
b = np.random.randn(1)[0] 

alpha = 0.0002 #Learning rate
its = 60000 #Número de iteraciones
stop = 0.0000001 #diferencia de error para detener las iteraciones
error = np.zeros((its,)) 

for i in range(its):
    # Actualización de parámetros con gradiente
    [m, b] = gradiente(m,b,alpha,x_train,y_train)
    # Cálculo de las predicción con el modelo de regresión
    y_pred = regresion(m,b,x_train)
    # Actualización del error
    error[i] = error_medio(y_train,y_pred)
    # Validación de diferencia de error para detener el gradiente 
    if error[i-1] and abs(error[i-1]-error[i])<=stop:
      break 
    
    # Impresión de resultados cada 1000 iteraciones
    if (i+1)%1000 == 0:
        print(" Iteración {}".format(i+1))
        print(" m: {:.1f}".format(m), "b: {:.1f}".format(b))
        print(" error: {:.3f}".format(error[i]))
        print("*------------------------*")
        

#Gráfica de error vs Iteraciones 
'''Podemos ver de forma gráfica la disminución del error conforme
el paso de las iteraciones'''
plt.plot(range(its),error,c='green')
plt.xlabel('Iteracion')
plt.ylabel('Error Cuadrático Medio')
plt.show()

#Gráfica de la recta obtenida en la regresión
plt.scatter(x_train,y_train,c='green')
plt.plot(x_train,y_pred,c='purple')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

#Prueba de predicciones 
print('-----Predicciones-------')
for i in x_test:
  presion = regresion(m,b,i)
  print('Edad:',i,'Presión: {:.2f}'.format(presion))

#Gráfica de resultados de la prueba 
y_pred=regresion(m,b,x_test)
plt.scatter(x_test,y_pred,marker='*',c='blue',label='Predicción')
plt.scatter(x_test,y_test,c='green',label='Real')
plt.plot(x_test,y_pred,'r--')
plt.xlabel('Edad')
plt.ylabel('Presión')
plt.legend()
plt.show()