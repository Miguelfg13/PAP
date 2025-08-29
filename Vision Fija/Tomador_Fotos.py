import cv2
import os

print("Directorio actual: ", os.getcwd())

# Ruta absoluta del directorio donde está el script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Crear la ruta a la carpeta "fotos" dentro del directorio del script
save_path = os.path.join(script_dir, "Fotos_Calibracion")
os.makedirs(save_path, exist_ok=True)

camweb = cv2.VideoCapture(1)

if not camweb.isOpened():
    print("No se puede abrir la cámara")
    exit()

photo_count = 0

print("Presiona 'SPACE' para tomar una foto, 'ESC' para salir.")

while True:
    ret, frame = camweb.read()

    if not ret:
        print("No se puede recibir imagen (stream end?). Saliendo ...")
        break

    # Mostrar el video en una ventana
    cv2.imshow('Presiona SPACE para tomar foto', frame)

    key = cv2.waitKey(1)
    
    if key == 27:  # ESC para salir
        break
    elif key == 32:  # SPACE para tomar foto
        photo_filename = os.path.join(save_path, f"foto_ajedrez_{photo_count}.jpg")
        cv2.imwrite(photo_filename, frame)
        print(f"Foto guardada como {photo_filename}")
        photo_count += 1

# Liberar la cámara y cerrar ventanas
camweb.release()
cv2.destroyAllWindows()