import cv2
import cv2.aruco as aruco
import numpy as np
import time
from Funciones_Codigos import obtener_Parametros_Camara, guardar_Datos_CSV

# ------ Funciones del código ------
def checar_Tiempo():
    return time.perf_counter() # Regresar el tiempo

def calcular_FrecuenciaEfectiva(N_validas, Tiempo):
    return N_validas/Tiempo

def sacar_Media_P05_P90(delta_T):
    delta_T_media = np.mean(delta_T)
    P05 = np.percentile(delta_T, 5)
    P95 = np.percentile(delta_T, 95)

    return delta_T_media, P05, P95

def sacar_Deltas_T(datos):
    tiempo = [fila["t"] for fila in datos]
    delta_T = np.diff(tiempo)

    datos[0]["delta_t"] = 0
    for i in range(1, len(datos)):
        datos[i]["delta_t"] = delta_T[i - 1]

    return sacar_Media_P05_P90(delta_T)



# ------ Configuración códigos ArUco ------
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# ------ Parametros camara ------
parametros_camara = obtener_Parametros_Camara()
resolucion_height = parametros_camara["resolucion_height"]
resolucion_width = parametros_camara["resolucion_width"]
matriz_camara = np.array(parametros_camara["matriz_camara"])
coeficientes_distorsion = np.array(parametros_camara["coef_dist"])


# ------ Parametros iniciales ------
marker_length = 0.143  # en metros
ventana_Tiempo = 60
cnt_Total_ArUcos = 0 # Cuantas veces se detectaron los códigos ArUco
cnt_ArUco_Carro = 0 # Cuantes veces se detecto el código ArUco del carro
codigo_ArUco_Carro = 4 # Código ArUco relacionado con el carro
timestamp = 0

datos_CSV = [] # Lista donde se guardaran la información (t, x, y)

camara = cv2.VideoCapture(1)

camara.set(cv2.CAP_PROP_FRAME_HEIGHT, resolucion_height)
camara.set(cv2.CAP_PROP_FRAME_WIDTH, resolucion_width)

tic = checar_Tiempo()
while True:
    ret, frame = camara.read()
    if not ret:
        break
    
    fps = camara.get(cv2.CAP_PROP_FPS)

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
                cnt_Total_ArUcos += 1
            
           
            if ids[i][0] == codigo_ArUco_Carro:
                timestamp = checar_Tiempo() - tic
                cnt_ArUco_Carro += 1
                datos_CSV.append({
                    "t": timestamp,
                    "id": int(ids[i][0]),
                    "x": float(x),
                    "y": float(y),
                    "z": float(z)
                })

            print(f"ID: {ids[i][0]} | Posición (x, y, z): ({x:.2f}, {y:.2f}, {z:.2f}) m | Distancia: {distancia:.2f} m | Tiempo: {timestamp:.4f}")

    cv2.imshow("Detección ArUco", frame)
    #cv2.waitKey(100)

    tac = checar_Tiempo()

    if tac - tic >= ventana_Tiempo:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camara.release()
cv2.destroyAllWindows()

print("En 60 seg. se detectaron los códigos ", cnt_Total_ArUcos, " veces.")
print("En 60 seg. se detecto: ", cnt_ArUco_Carro, "veces.")
print("Frecuencia efectiva: ", calcular_FrecuenciaEfectiva(cnt_ArUco_Carro, ventana_Tiempo))
dropout = ((fps * ventana_Tiempo) - cnt_ArUco_Carro)/(fps * ventana_Tiempo)
print(f"Los FPS de la camara son: {fps}, con esto calculamos el dropout: {dropout * 100}")
delta_T_media, P05, P95 = sacar_Deltas_T(datos_CSV)
print(f"La media de la Delta T es: {delta_T_media}.\n"
      f"El P05: {P05}. \n"
      f"El P95: {P95}" )
guardar_Datos_CSV(datos_CSV)
print("Se han guardado los datos en el CSV.")
