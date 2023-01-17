import cv2 as cv
import os
import numpy as np
from time import time

data= "C:/Users/abela/Documents/ProyectosPython/Reconocimiento de imagenes/Reconocimiento facial/Data"
listaData = os.listdir(data)
ids = []
rostrosData=[]
id=0
tinicial = time()
for i in listaData:
    rutaCompleta = data + "/" + i
    print(f"Iniciando lectura {i}...")
    for archivo in os.listdir(rutaCompleta):
        print('Imagenes: ', i +'/'+archivo)
        ids.append(id)
        rostrosData.append(cv.imread(rutaCompleta + "/" + archivo, 0))
    id+=1
    tfinal=time()
    tiempoTotal = tfinal-tinicial
    print('tiempo total de lectura: ', tiempoTotal)

entrenamientoM1 = cv.face.EigenFaceRecognizer_create()
print("Iniciando el entrenamiento....")

entrenamientoM1.train(rostrosData, np.array(ids))
tfentrenamiento = time()
ttotalentre = tfentrenamiento - tiempoTotal
entrenamientoM1.write("EntrenamientoEigenFaceRecognizer.xml")
print("Entrenamiento concluido")
print('Tiempo total del entrenamiento: ', ttotalentre)

