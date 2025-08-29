import cv2
import cv2.aruco as aruco
import numpy as np

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# ⚠️ Reemplazá estos con tus valores reales de calibración:
camera_matrix = np.array([[700.27703336, 0, 344.1757582 ]
                          [0, 704.72638774, 226.15187565]
                          [0,  0, 1]], dtype=np.float32)

dist_coeffs = np.array([[-0.44453011, 0.20908467, 0.00520405, -0.00129481, 0.9669697]])  # o con ceros si no tenés distorsión

marker_length = 0.05  # en metros (5 cm)

camara = cv2.VideoCapture(1)

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
            aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

            # Obtener coordenadas x, y, z del marcador respecto a la cámara
            x, y, z = tvec[0]

            # Calcular la distancia Euclidiana
            distancia = np.linalg.norm(tvec)

            print(f"ID: {ids[i][0]} | Posición (x, y, z): ({x:.2f}, {y:.2f}, {z:.2f}) m | Distancia: {distancia:.2f} m")

    cv2.imshow("Detección ArUco", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camara.release()
cv2.destroyAllWindows()

