import cv2
import cv2.aruco as aruco
import csv
import numpy as np
from Funciones_Codigos import obtener_Direccion_Carpeta_Absoluta, juntar_Direccion_Archivo

ruta_video = juntar_Direccion_Archivo(obtener_Direccion_Carpeta_Absoluta(), 'videoSalida.avi')

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# ------- Valores ------
rejilla_ids = [1, 2, 3, 5]
rejilla_coords = {
    1: (0.0, 0.0),
    2: (2.554, 0.0),
    3: (0.0, 1.803),
    5: (2.554, 1.803)
}
frente_id = 23
atras_id = 24


video = cv2.VideoCapture(ruta_video)

fps = video.get(cv2.CAP_PROP_FPS)
print("FPS del video:", fps)

# Archivo CSV de salida
csv_filename = "trayectoria_robot.csv"
ruta_trayectoria = juntar_Direccion_Archivo(obtener_Direccion_Carpeta_Absoluta(), csv_filename)
csv_file = open(ruta_trayectoria, mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["t", "x", "y", "theta", "id_front", "id_back"])

# ==============================
# FUNCIONES
# ==============================
def get_center(corners):
    """Calcula el centro (x,y) de un marcador dado sus esquinas"""
    c = corners[0]
    x = np.mean(c[:, 0])
    y = np.mean(c[:, 1])
    return (x, y)

# ==============================
# PROCESAMIENTO FRAME A FRAME
# ==============================
frame_idx = 0
H = None  # homografía (se calcula al inicio con la rejilla)

while True:
    ret, imagen = video.read()
    if not ret:
        print("No se detecto la imagen.")
        break
    
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        ids = ids.flatten()

        # ---- 1. Detectar los 4 ArUcos de la rejilla ----
        pts_pix = []
        pts_real = []
        for i, id in enumerate(ids):
            if id in rejilla_ids:
                cx, cy = get_center(corners[i])
                pts_pix.append([cx, cy])
                pts_real.append(rejilla_coords[id])

        if len(pts_pix) == 4:  # si se detectaron los 4
            pts_pix = np.array(pts_pix, dtype=np.float32)
            pts_real = np.array(pts_real, dtype=np.float32)
            H, _ = cv2.findHomography(pts_pix, pts_real)

        # ---- 2. Detectar los ArUcos del robot ----
        if H is not None and frente_id in ids and atras_id in ids:
            # Frente
            idx_f = np.where(ids == frente_id)[0][0]
            cx_f, cy_f = get_center(corners[idx_f])
            pt_f = np.array([[[cx_f, cy_f]]], dtype=np.float32)
            real_f = cv2.perspectiveTransform(pt_f, H)[0][0]

            # Trasero
            idx_b = np.where(ids == atras_id)[0][0]
            cx_b, cy_b = get_center(corners[idx_b])
            pt_b = np.array([[[cx_b, cy_b]]], dtype=np.float32)
            real_b = cv2.perspectiveTransform(pt_b, H)[0][0]

            # ---- 3. Calcular posición y orientación ----
            x = (real_f[0] + real_b[0]) / 2
            y = (real_f[1] + real_b[1]) / 2
            theta = np.degrees(np.arctan2(real_f[1] - real_b[1],
                                          real_f[0] - real_b[0]))

            # ---- 4. Calcular tiempo ----
            t = frame_idx / fps

            # ---- 5. Guardar en CSV ----
            csv_writer.writerow([t, x, y, theta, frente_id, atras_id])

            # ---- 6. Visualización ----
            cv2.circle(imagen, (int(cx_f), int(cy_f)), 5, (0, 0, 255), -1)
            cv2.circle(imagen, (int(cx_b), int(cy_b)), 5, (255, 0, 0), -1)
            cv2.arrowedLine(imagen,
                            (int(cx_b), int(cy_b)),
                            (int(cx_f), int(cy_f)),
                            (0, 255, 0), 2)

            cv2.putText(imagen, f"x={x:.1f}, y={y:.1f}, th={theta:.1f}",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Mostrar el frame
    cv2.imshow("Deteccion ArUco", imagen)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_idx += 1

video.release()
csv_file.close()
cv2.destroyAllWindows()
print("Datos guardados en:", csv_filename)
