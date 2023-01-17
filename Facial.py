import cv2 as cv
import os
import imutils

modelo = "" #añadir el nombre de la persona a tomar las fotos
ruta = "C:/Users/abela/Documents/ProyectosPython/Reconocimiento de imagenes/Reconocimiento facial"
ruta_completa = ruta + "/" + modelo
if not os.path.exists(ruta_completa):
    os.makedirs(ruta_completa)

cap = cv.VideoCapture(0)#Captura de video
face_cascade = cv.CascadeClassifier(r"C:\Users\abela\Documents\ProyectosPython\Reconocimiento de imagenes\Reconocimiento facial\haarcascade_frontalface_default.xml")

id = 0
while True:
    respuesta, img = cap.read()
    if respuesta == False: break
    img = imutils.resize(img, width=640) #Reducir la calidad de la imagen para ahorro de memoria

    gray = cv.cvtColor(img, 10)
    idcaptura = img.copy() 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x,y), (x+w,y+h), (0, 0, 255), 5)
        capturado = idcaptura[y:y+w, x:x+h]
        capturado=cv.resize(capturado, (160, 160), interpolation=cv.INTER_CUBIC)
        cv.imwrite(ruta_completa + '/imagen_{}.jpg'.format(id), capturado)
        id = id + 1
    #cv.imshow("Imagen final", img)
    if id == 351:
        break

cap.release
cv.destroyAllWindows()
