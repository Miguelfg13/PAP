import cv2
import os
from Funciones_Codigos import obtener_Direccion_Carpeta_Absoluta, juntar_Direccion_Archivo

# Crear la ruta a la carpeta "fotos" dentro del directorio del script
dir_carpeta_fotos = juntar_Direccion_Archivo(
    obtener_Direccion_Carpeta_Absoluta(), "Fotos_Calibracion"
    )
os.makedirs(dir_carpeta_fotos, exist_ok=True)

camara_web = cv2.VideoCapture(1)

camara_web.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
camara_web.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

if not camara_web.isOpened():
    print("No se puede abrir la cámara")
    exit()

contador_fotos = 0

print("Presiona 'SPACE' para tomar una foto, 'ESC' para salir.")

while True:
    ret, imagen = camara_web.read()

    if not ret:
        print("No se puede recibir imagen (stream end?). Saliendo ...")
        break

    # Mostrar el video en una ventana
    cv2.imshow('Presiona SPACE para tomar foto', imagen)

    key = cv2.waitKey(1)
    
    if key == 27:  # ESC para salir
        break
    elif key == 32:  # SPACE para tomar foto
        nombre_foto = juntar_Direccion_Archivo(
            dir_carpeta_fotos, f"foto_ajedrez_{contador_fotos}.jpg"
            )
        cv2.imwrite(nombre_foto, imagen)
        print(f"Foto guardada como {nombre_foto}")
        contador_fotos += 1

# Liberar la cámara y cerrar ventanas
camara_web.release()
cv2.destroyAllWindows()