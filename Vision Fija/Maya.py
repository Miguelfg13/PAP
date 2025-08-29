import cv2
import numpy as np

# ---------------------------
# Configuración ArUco y cámara
# ---------------------------
aruco_dict  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters  = cv2.aruco.DetectorParameters()
detector    = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Ajusta a tu cámara real si ya la calibraste
camera_matrix = np.array([[800,   0, 320],
                          [  0, 800, 240],
                          [  0,   0,   1]], dtype=np.float32)
dist_coeffs   = np.zeros((5, 1), dtype=np.float32)

# IDs
corner_ids = [1, 2, 5, 3]   # cuatro esquinas (en cualquier orden)
robot_id   = 4              # marcador del robot

# Tamaño de la cuadrícula (N x N)
GRID_N = 10

# --------------------------------
# Utilidades para homografía y malla
# --------------------------------
def order_quad(pts):
    """
    Ordena 4 puntos como TL, TR, BR, BL según su posición en la imagen.
    pts: iterable de 4 puntos (x, y)
    """
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)          # x + y
    d = np.diff(pts, axis=1)     # x - y
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def draw_grid_homography(img, H, N=10, color=(0,255,0), thickness=1):
    """
    Dibuja una malla NxN mapeando líneas del cuadrado unitario (u,v)∈[0,1]^2
    hacia la imagen mediante la homografía H.
    """
    # Verticales: u = i/N
    for i in range(N+1):
        u = i / float(N)
        src = np.array([[[u, 0.0]],
                        [[u, 1.0]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, H)
        p1 = tuple(dst[0,0].astype(int))
        p2 = tuple(dst[1,0].astype(int))
        cv2.line(img, p1, p2, color, thickness)

    # Horizontales: v = i/N
    for i in range(N+1):
        v = i / float(N)
        src = np.array([[[0.0, v]],
                        [[1.0, v]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, H)
        p1 = tuple(dst[0,0].astype(int))
        p2 = tuple(dst[1,0].astype(int))
        cv2.line(img, p1, p2, color, thickness)

def pix_to_grid_uv(pt_pix, H_inv):
    """
    Convierte un punto de imagen (x,y) a coordenadas (u,v) del cuadrado unitario
    usando la homografía inversa H_inv.
    """
    src = np.array([[[pt_pix[0], pt_pix[1]]]], dtype=np.float32)
    uv = cv2.perspectiveTransform(src, H_inv)[0,0]
    return float(uv[0]), float(uv[1])

# -----------------
# Loop de la cámara
# -----------------
cap = cv2.VideoCapture(1)

# Homografía inversa disponible en el frame actual
current_H_inv = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detección de ArUco
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        ids = ids.flatten()
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # (opcional) pose para ejes 3D de cada marcador
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, 0.05, camera_matrix, dist_coeffs)

        for i in range(len(ids)):
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                              rvecs[i], tvecs[i], 0.03)

        # -----------------------------
        # Cuadrícula por homografía 2D
        # -----------------------------
        centers_px = {}
        for i, mid in enumerate(ids):
            if mid in corner_ids:
                c = corners[i][0]              # (4,2)
                center = c.mean(axis=0)        # (2,)
                centers_px[int(mid)] = center

        if len(centers_px) == 4:
            # Orden TL, TR, BR, BL
            pts_dst = order_quad([centers_px[k] for k in centers_px.keys()])

            # Cuadrado unitario
            pts_src = np.array([[0.,0.],
                                [1.,0.],
                                [1.,1.],
                                [0.,1.]], dtype=np.float32)

            # Homografía del cuadrado unitario -> cuadrilátero
            H = cv2.getPerspectiveTransform(pts_src, pts_dst)
            current_H_inv = np.linalg.inv(H)

            # Dibuja la malla
            draw_grid_homography(frame, H, N=GRID_N, color=(0,255,0), thickness=1)
        else:
            current_H_inv = None

        # -------------------------------------------
        # Robot en coordenadas de CUADRÍCULA (0,0) BL
        # -------------------------------------------
        if (robot_id in ids) and (current_H_inv is not None):
            idx = list(ids).index(robot_id)
            c_robot = corners[idx][0]           # (4,2) pixeles del marcador

            # Centro en pixeles
            center_px = c_robot.mean(axis=0)
            px, py = int(center_px[0]), int(center_px[1])

            # (u,v) en [0,1]^2 con origen arriba-izq -> invertimos v
            u, v = pix_to_grid_uv((px, py), current_H_inv)
            u = np.clip(u, 0.0, 1.0)
            v = 1.0 - np.clip(v, 0.0, 1.0)      # origen abajo-izquierda

            # Coordenadas en celdas (0..GRID_N)
            gx = u * GRID_N
            gy = v * GRID_N

            # Ángulo relativo a la malla:
            # Tomamos una dirección sobre el tag (esquina 0 -> 1) y la llevamos a (u,v),
            # también invirtiendo v para mantener el mismo sistema.
            p_dir_img = (c_robot[0] + 0.5*(c_robot[1] - c_robot[0]))  # un punto hacia +x del tag
            u0, vv0 = pix_to_grid_uv(center_px, current_H_inv)
            u1, vv1 = pix_to_grid_uv(p_dir_img, current_H_inv)
            v0 = 1.0 - vv0
            v1 = 1.0 - vv1
            du, dv = (u1 - u0), (v1 - v0)
            angle_grid = np.degrees(np.arctan2(dv, du))  # 0° = +X de la malla (derecha), CCW

            # Dibujo/Texto
            cv2.circle(frame, (px, py), 6, (0, 0, 255), -1)
            cv2.putText(frame, f"Robot ({gx:.2f}, {gy:.2f}) Ang_grid: {angle_grid:.1f} deg",
                        (px+10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # Mostrar
    cv2.imshow("Cuadricula ArUco", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
