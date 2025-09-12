import cv2
import cv2.aruco as aruco
import numpy as np
import time
import json
import os
import pandas as pd

# Se obtienen los valores de los paramatros de la camara del archivo json.
def obtener_parametros():
    with open("parametros_camara.json", "r") as f:
        parametros = json.load(f)
    
    return parametros

# Función para guardar los valores obtenidos (delta t, x, y, z) en un archivo csv.
def guardar_en_CSV(datos):
    direccion_carpeta = os.path.dirname(os.path.abspath(__file__))
    dir = os.path.join(direccion_carpeta, "valores.csv")

    df = pd.DataFrame(datos)
    df.to_csv(dir, index = False)

# Configuración códigos ArUco
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# Parametros camara
parametros_cam = obtener_parametros()
matriz_camara = np.array[parametros_cam["matriz_camara"]]
coeficientes_distorsion = np.array[parametros_cam["coef_dist"]]
marker_length = 0.05  # en metros (5 cm)

# Parametros iniciales:
ventana_Tiempo = 60
cnt_Total_ArUcos = 0 # Cuantas veces se detectaron los códigos ArUco
cnt_ArUco_Carro = 0 # Cuantes veces se detecto el código ArUco del carro
codigo_ArUco_Carro = 4 # Código ArUco relacionado con el carro

datos_CSV = [] # Lista donde se guardaran la información (t, x, y)

# Funciones:
def checar_Tiempo():
    return time.perf_counter() # Regresar el tiempo

def calcular_FrecuenciaEfectiva(N_validas, Tiempo):
    return N_validas/Tiempo

camara = cv2.VideoCapture(0)
tic = checar_Tiempo()
while True:
    ret, frame = camara.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimamos la pose
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, marker_length, matriz_camara, coeficientes_distorsion
        )

        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            cv2.drawFrameAxes(frame, matriz_camara, coeficientes_distorsion, rvec, tvec, 0.05)

            # Obtener coordenadas x, y, z del marcador respecto a la cámara
            x, y, z = tvec[0]

            # Calcular la distancia Euclidiana
            distancia = np.linalg.norm(tvec)

            if len(ids) >= 5:
                cnt_Total_ArUcos =+ 1
            
            timestamp = checar_Tiempo() - tic
            if ids[i][0] == codigo_ArUco_Carro:
                cnt_ArUco_Carro =+ 1
                datos_CSV.append({
                    "t": timestamp,
                    "id": int(ids[i][0]),
                    "x": float(x),
                    "y": float(y),
                    "z": float(z)
                })

            #print(f"ID: {ids[i][0]} | Posición (x, y, z): ({x:.2f}, {y:.2f}, {z:.2f}) m | Distancia: {distancia:.2f} m")

    cv2.imshow("Detección ArUco", frame)
    cv2.waitKey(100)

    tac = checar_Tiempo()

    if tac - tic >= ventana_Tiempo:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camara.release()
cv2.destroyAllWindows()

print("En 60 seg. se detectaron los códigos ", cnt_Total_ArUcos, " veces.")
print("En 60 seg. se detecto: ", cnt_ArUco_Carro)
print("Frecuencia efectiva: ", calcular_FrecuenciaEfectiva(cnt_ArUco_Carro, ventana_Tiempo))
