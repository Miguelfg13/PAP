import cv2
import cv2.aruco as aruco
import numpy as np
import json
import os
from Funciones_Codigos import obtener_Parametros_Camara

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

parametros_camara = obtener_Parametros_Camara()
resolucion_height = parametros_camara["resolucion_height"]
resolucion_width = parametros_camara["resolucion_width"]
camera_matrix = np.array(parametros_camara["matriz_camara"])
dist_coeffs = np.array(parametros_camara["coef_dist"])
marker_length = 0.143

camara = cv2.VideoCapture(1)

camara.set(cv2.CAP_PROP_FRAME_HEIGHT, resolucion_height)
camara.set(cv2.CAP_PROP_FRAME_WIDTH, resolucion_width)

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
            corners, marker_length, camera_matrix, dist_coeffs
        )

        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

            # Obtener coordenadas x, y, z del marcador respecto a la cámara
            x, y, z = tvec[0]

            # Calcular la distancia Euclidiana
            distancia = np.linalg.norm(tvec)

            print(f"ID: {ids[i][0]} | Posición (x, y, z): ({x:.2f}, {y:.2f}, {z:.2f}) m | Distancia: {distancia:.2f} m")

    cv2.imshow("Detección ArUco", frame)

    cv2.waitKey(100)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camara.release()
cv2.destroyAllWindows()

