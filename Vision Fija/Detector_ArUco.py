import cv2
import cv2.aruco as aruco

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

camara = cv2.VideoCapture(1)

# width = camara.get(cv2.CAP_PROP_FRAME_WIDTH)
# height = camara.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print(f"Resolución inicial: {int(width)} x {int(height)}")

camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

width = camara.get(cv2.CAP_PROP_FRAME_WIDTH)
height = camara.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Resolución despues de setear 1280x720: {int(width)} x {int(height)}")


while True:
    ret, frame = camara.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        for i, corners in zip(ids, corners):
            print(f"Detectado ID: {i}")

    cv2.imshow("Detección ArUco", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camara.release()
cv2.destroyAllWindows()
