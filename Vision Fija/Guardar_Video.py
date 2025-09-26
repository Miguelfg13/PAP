import cv2
from Funciones_Codigos import obtener_Direccion_Carpeta_Absoluta, juntar_Direccion_Archivo

ruta_video = juntar_Direccion_Archivo(obtener_Direccion_Carpeta_Absoluta(), 'video_RealvsCodigo.avi')

camara_web = cv2.VideoCapture(1)
video_salida = cv2.VideoWriter(ruta_video, cv2.VideoWriter_fourcc(*'XVID'), 30.0, (1280, 720))

while True:
    ret, imagen = camara_web.read()
    if not ret:
        print("No se detecto la imagen.")
        break
    
    cv2.imshow('video', imagen)
    video_salida.write(imagen)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camara_web.release()
video_salida.release()
cv2.destroyAllWindows()