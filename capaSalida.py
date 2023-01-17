import cv2 as cv
import os

data = "C:/Users/abela/Documents/ProyectosPython/Reconocimiento de imagenes/Reconocimiento facial/Data"
listaData = os.listdir(data)
entrenamientoEigen = cv.face.EigenFaceRecognizer_create()
entrenamientoEigen.read(r"C:\Users\abela\Documents\ProyectosPython\Reconocimiento de imagenes\Reconocimiento facial\EntrenamientoEigenFaceRecognizer.xml")
ruidos = cv.CascadeClassifier(r"C:\Users\abela\Documents\ProyectosPython\Reconocimiento de imagenes\Reconocimiento facial\haarcascade_frontalface_default.xml")

#Captura de camara
camara = cv.VideoCapture(1)
while True:
    _,img = camara.read()
    gray = cv.cvtColor(img, 10)
    idcaptura = gray.copy()
    cara = ruidos.detectMultiScale(gray, 1.3,5)
    for (x, y, w, h) in cara:
        #cv.rectangle(img, (x,y), (x+w,y+h), (0, 0, 255), 5)
        capturado = idcaptura[y:y+w, x:x+h]
        capturado=cv.resize(capturado, (160, 160), interpolation=cv.INTER_CUBIC)
        resultado = entrenamientoEigen.predict(capturado)
        cv.putText(img, f"{resultado}", (x,y-5), 1,1.3, (0,0,255),1, cv.LINE_AA)
        if resultado[1] < 8400:
            cv.putText(img, f"{listaData[resultado[0]]}", (x,y-20), 1,2, (0,255,0),1, cv.LINE_AA)
            cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0),3)
        else:
            cv.putText(img, "No encontrado", (x,y-20), 1,1.5, (0,0,255),1, cv.LINE_AA)
            cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0),3)

    cv.imshow("Imagen final", img)

    if cv.waitKey(1)==ord("q"):
        break

camara.release()
cv.destroyAllWindows()
