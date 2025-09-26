import cv2
import cv2.aruco as aruco
import numpy as np
import os
from Funciones_Codigos import obtener_Parametros_Camara, gurdar_Parametros_Extrinsecos_Camara

# ------ Configuración códigos ArUco ------
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# ------ Obtencion de los parametros de la camara ------
paremetros_camara = obtener_Parametros_Camara()
resolucion_height = paremetros_camara["resolucion_height"]
resolucion_width = paremetros_camara["resolucion_width"]
matriz_camara = np.array(paremetros_camara["matriz_camara"])
coef_distorsion = np.array(paremetros_camara["coef_dist"])

# ------ Parametros generales ------
tamaño_aruco = 0.143  # en metros
coordenadas_world = {
    1: np.array([0.0, 0.0, 0.0]),   #Esquina inferior izq.
    2: np.array([2.554, 0.0, 0.0]),   #Esquina inferior der.
    3: np.array([0.0, 1.803, 0]),    #Esquina superior izq.
    5: np.array([2.554, 1.803, 0])     #Esquina superior der.
}
obj_points = [] # Puntos 3D conocidos
img_points = []  # Puntos 2D dectedados en la imagen

camara = cv2.VideoCapture(1)
camara.set(cv2.CAP_PROP_FRAME_HEIGHT, resolucion_height)
camara.set(cv2.CAP_PROP_FRAME_WIDTH, resolucion_width)

ret, imagen = camara.read()
if not ret:
    print("No se pudo tomar una imagen.")
    exit()

corners, ids, rejected = detector.detectMarkers(imagen)
if ids is None:
    print("No se detectaron ArUcos en la imagen.")
    exit()

for i, marker_id in enumerate(ids.flatten()):
    if marker_id in coordenadas_world:
        c = corners[i][0]
        centro = c.mean(axis=0)
        img_points.append(centro)
        obj_points.append(coordenadas_world[marker_id])

obj_points = np.array(obj_points, dtype=np.float32)
img_points = np.array(img_points, dtype=np.float32)

ret, rvec, tvec = cv2.solvePnP(obj_points, img_points, matriz_camara, coef_distorsion)

R, _ = cv2.Rodrigues(rvec)

print("Matriz de rotación (R):\n", R)
print("Vector de traslación (t):\n", tvec)
gurdar_Parametros_Extrinsecos_Camara(R, tvec)
print("Se guardaron la R y t en el archivo json")


# --- Dibujar ejes en un ArUco de referencia ---
axis = np.float32([[0.1,0,0], [0,0.1,0], [0,0,-0.1]])  # ejes de 20 cm
imgpts, _ = cv2.projectPoints(axis, rvec, tvec, matriz_camara, coef_distorsion)

corner = tuple(img_points[0].astype(int))  # primer punto detectado
imagen = cv2.line(imagen, corner, tuple(imgpts[0].ravel().astype(int)), (0,0,255), 3) # X
imagen = cv2.line(imagen, corner, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 3) # Y
imagen = cv2.line(imagen, corner, tuple(imgpts[2].ravel().astype(int)), (255,0,0), 3) # Z

cv2.imshow("Extrinseca con 4 ArUco", imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()

