import cv2
import cv2.aruco as aruco
import Funciones_Codigos as fc

parametros_camara = fc.obtener_Parametros_Camara()

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

camara = cv2.VideoCapture(1)

camara.set(cv2.CAP_PROP_FRAME_WIDTH, parametros_camara["resolucion_width"])
camara.set(cv2.CAP_PROP_FRAME_HEIGHT, parametros_camara["resolucion_height"])

while True:
    ret, imagen = camara.read()
    if not ret:
        break

    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    esquinas, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        aruco.drawDetectedMarkers(imagen, esquinas, ids)
        for i, esquinas in zip(ids, esquinas):
            print(f"Detectado ID: {i}")

    cv2.imshow("Detección ArUco", imagen)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camara.release()
cv2.destroyAllWindows()
