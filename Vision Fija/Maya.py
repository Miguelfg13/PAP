import cv2
import numpy as np

# ==========================
#  Apertura de cámara Win)
# ==========================
def open_camera(index=2, width=1920, height=1080, fps=30, prefer_mjpg=True):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if not cap.isOpened():
        raise RuntimeError("No pude abrir la cámara en ningún backend.")

    if prefer_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    afps = cap.get(cv2.CAP_PROP_FPS)

    # Reintentos si no quedó en 1080p
    if (aw, ah) != (width, height):
        # prueba YUY2
        if prefer_mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUY2'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS,          fps)
            aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            afps = cap.get(cv2.CAP_PROP_FPS)

    if (aw, ah) != (width, height):
        # fallback 720p
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS,          fps)
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        afps = cap.get(cv2.CAP_PROP_FPS)

    print(f"[INFO] Resolución activa: {aw}x{ah} @ {afps:.1f} FPS")
    return cap

# ==========================
#  ArUco + Homografía
# ==========================
aruco_dict  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters  = cv2.aruco.DetectorParameters()
detector    = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Sustituye por tu calibración real si la tienes
camera_matrix = np.array([[800,   0, 320],
                          [  0, 800, 240],
                          [  0,   0,   1]], dtype=np.float32)
dist_coeffs   = np.zeros((5, 1), dtype=np.float32)

corner_ids = [1, 2, 5, 3]   # ids de las 4 esquinas (en cualquier orden)
robot_id   = 4              # id del robot
GRID_N     = 10             # malla NxN

def order_quad(pts):
    """Ordena 4 puntos como TL, TR, BR, BL según posición en imagen."""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)          # x+y
    d = np.diff(pts, axis=1)     # x-y
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def draw_grid_homography(img, H, N=10, color=(0,255,0), thickness=1):
    """Dibuja malla NxN mapeando líneas del cuadrado unitario via H."""
    for i in range(N+1):
        u = i / float(N)
        src = np.array([[[u, 0.0]], [[u, 1.0]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, H)
        cv2.line(img, tuple(dst[0,0].astype(int)), tuple(dst[1,0].astype(int)), color, thickness)
    for i in range(N+1):
        v = i / float(N)
        src = np.array([[[0.0, v]], [[1.0, v]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, H)
        cv2.line(img, tuple(dst[0,0].astype(int)), tuple(dst[1,0].astype(int)), color, thickness)

def pix_to_grid_uv(pt_pix, H_inv):
    """(x,y) imagen → (u,v) en [0,1]^2 via H_inv."""
    src = np.array([[[pt_pix[0], pt_pix[1]]]], dtype=np.float32)
    uv = cv2.perspectiveTransform(src, H_inv)[0,0]
    return float(uv[0]), float(uv[1])

# ==========================
#  Main
# ==========================
def main():
    cap = open_camera(index=1, width=1920, height=1080, fps=30, prefer_mjpg=True)

    cv2.namedWindow("Cuadricula ArUco 1080p", cv2.WINDOW_NORMAL)
    DISPLAY_SCALE = 0.75  # zoom de la ventana (solo visualización)

    misses = 0
    MAX_MISSES = 30
    current_H_inv = None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            misses += 1
            if misses > MAX_MISSES:
                print("[WARN] Reintentando abrir cámara…")
                cap.release()
                cap = open_camera(index=0, width=1920, height=1080, fps=30, prefer_mjpg=True)
                misses = 0
            continue
        misses = 0

        # Detección
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is not None:
            ids = ids.flatten()
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # (opcional) ejes 3D de cada marcador
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, 0.05, camera_matrix, dist_coeffs)
            for i in range(len(ids)):
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                                  rvecs[i], tvecs[i], 0.03)

            # Homografía de las 4 esquinas (usando centros de marcadores)
            centers_px = {}
            for i, mid in enumerate(ids):
                if mid in corner_ids:
                    c = corners[i][0]          # (4,2)
                    centers_px[int(mid)] = c.mean(axis=0)

            if len(centers_px) == 4:
                pts_dst = order_quad([centers_px[k] for k in centers_px.keys()])  # TL,TR,BR,BL
                pts_src = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]], dtype=np.float32)
                H = cv2.getPerspectiveTransform(pts_src, pts_dst)
                current_H_inv = np.linalg.inv(H)
                draw_grid_homography(frame, H, N=GRID_N, color=(0,255,0), thickness=1)
            else:
                current_H_inv = None

            # Robot en coords de malla (origen 0,0 abajo-izq)
            if (robot_id in ids) and (current_H_inv is not None):
                idx = list(ids).index(robot_id)
                c_robot = corners[idx][0]

                center_px = c_robot.mean(axis=0)
                px, py = int(center_px[0]), int(center_px[1])

                # (u,v) con origen arriba-izq -> invertimos v para abajo-izq
                u, v_top = pix_to_grid_uv((px, py), current_H_inv)
                u = np.clip(u, 0.0, 1.0)
                v = 1.0 - np.clip(v_top, 0.0, 1.0)

                gx = u * GRID_N
                gy = v * GRID_N

                # Ángulo relativo a la malla (0° hacia +X, CCW)
                p_dir_img = (c_robot[0] + 0.5*(c_robot[1] - c_robot[0]))
                u0, v0_top = pix_to_grid_uv(center_px, current_H_inv)
                u1, v1_top = pix_to_grid_uv(p_dir_img, current_H_inv)
                v0 = 1.0 - v0_top
                v1 = 1.0 - v1_top
                du, dv = (u1 - u0), (v1 - v0)
                angle_grid = np.degrees(np.arctan2(dv, du))

                cv2.circle(frame, (px, py), 6, (0,0,255), -1)
                cv2.putText(frame, f"Robot ({gx:.2f}, {gy:.2f}) Ang_grid: {angle_grid:.1f} deg",
                            (px+10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # --------- Display escalado para que quepa en pantalla ---------
        disp = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                          interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Cuadricula ArUco 1080p", disp)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:    # ESC
            break
        elif k in (ord('+'), ord('=')):   # zoom in
            DISPLAY_SCALE = min(1.0, DISPLAY_SCALE + 0.05)
        elif k in (ord('-'), ord('_')):   # zoom out
            DISPLAY_SCALE = max(0.30, DISPLAY_SCALE - 0.05)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
