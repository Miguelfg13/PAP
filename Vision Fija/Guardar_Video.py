import cv2
from Funciones_Codigos import obtener_Direccion_Carpeta_Absoluta, juntar_Direccion_Archivo, obtener_Parametros_Camara

ruta_video = juntar_Direccion_Archivo(obtener_Direccion_Carpeta_Absoluta(), 'video_RealvsCodigo.mp4')

parametros_camara = obtener_Parametros_Camara()
resolucion_height = parametros_camara["resolucion_height"]
resolucion_width = parametros_camara["resolucion_width"]

camara_web = cv2.VideoCapture(0)
video_salida = cv2.VideoWriter(ruta_video, cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (1280, 720))

camara_web.set(cv2.CAP_PROP_FRAME_HEIGHT, resolucion_height)
camara_web.set(cv2.CAP_PROP_FRAME_WIDTH, resolucion_width)

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