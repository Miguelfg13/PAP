import cv2
import numpy as np
import os
from datetime import date
from Funciones_Codigos import guardar_Parametros_Camara, obtener_Direccion_Carpeta_Absoluta, juntar_Direccion_Archivo

images_path = juntar_Direccion_Archivo(
    obtener_Direccion_Carpeta_Absoluta(), "Fotos_Calibracion"
    )

# Tamaño del patrón: número de esquinas internas (no cuadros)
pattern_size = (7, 7)
square_size = 0.024  # Tamaño real de cada cuadrado (en metros)

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

print("Matriz de cámara (intrínseca):\n", camera_matrix)
print("Coeficientes de distorsión:\n", dist_coeffs)
print("Reporjection error: ", ret)

guardar_Parametros_Camara("1280x720", camera_matrix, dist_coeffs, date.today())
print("Se guardaron los parametros en el archivo json")
