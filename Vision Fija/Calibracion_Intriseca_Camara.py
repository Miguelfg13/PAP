import cv2
import numpy as np
import os
import json
from datetime import date

# Función para guardar los datos de la calibración
def guardar_Datos(direccion_carpeta, matriz_camara, coeficientes_distorsion, fecha_calibracion):
    data = {
        "matriz_camara": matriz_camara.tolist(),
        "coef_dist": coeficientes_distorsion.tolist(),
        "fecha_calibracion": str(fecha_calibracion)
    }

    dir = os.path.join(direccion_carpeta, "parametros_camara.json")

    with open(dir, "w") as f:
        json.dump(data, f, indent=4)

# Ruta absoluta del directorio donde está el script
script_dir = os.path.dirname(os.path.abspath(__file__))
images_path = os.path.join(script_dir, "Fotos_Calibracion") # fotos_ajedrez -> camara personal / Fotos_Calibracion -> camara laboratorio

# Tamaño del patrón: número de esquinas internas (no cuadros)
pattern_size = (7, 7)
square_size = 0.026  # Tamaño real de cada cuadrado (en metros)

# Preparar coordenadas del patrón (puntos 3D en el mundo real)
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size  # Escalar según el tamaño real de los cuadros

objpoints = []  # Puntos 3D en el mundo real
imgpoints = []  # Puntos 2D detectados en la imagen

# Filtrar archivos .jpg dentro de esa carpeta
images = [os.path.join(images_path, f)
          for f in os.listdir(images_path)
          if f.lower().endswith('.jpg')]

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Encontrar esquinas del tablero
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Dibujar las esquinas encontradas
        cv2.drawChessboardCorners(img, pattern_size, corners, ret)
        cv2.imshow('Detección', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# Calibrar la cámara
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

guardar_Datos(script_dir, camera_matrix, dist_coeffs, date.today())
print("Matriz de cámara (intrínseca):\n", camera_matrix)
print("Coeficientes de distorsión:\n", dist_coeffs)
print("Reporjection error: ", ret)
