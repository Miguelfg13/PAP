import cv2
import numpy as np
import math
import time

# ==========================
#  Modo de prueba con imagen
# ==========================
USE_STATIC_IMAGE = True
IMAGE_PATH       = "aruco_test_scene.png"   # <- cambia si tu archivo se llama distinto
TARGET_W, TARGET_H = 1920, 1080             # resolución de trabajo de tu app

# ==========================
#  Apertura de cámara (Win)
# ==========================
def open_camera(index=1, width=1920, height=1080, fps=30, prefer_mjpg=True):
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
#  Fuente de frames unificada
# ==========================
class FrameSource:
    """
    Provee .read() y .release() sea desde cámara o imagen estática.
    En modo imagen, devuelve copias del mismo frame para mantener el pipeline.
    """
    def __init__(self, use_static_image, image_path, width, height, fps=30, cam_index=1):
        self.use_static = use_static_image
        self.fps_delay = 1.0 / max(1, fps)
        self._last_time = 0.0
        self.cap = None
        self.frame_static = None
        if self.use_static:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"No pude leer la imagen: {image_path}")
            # redimensiona si es necesario para igualar tu pipeline
            if (img.shape[1], img.shape[0]) != (width, height):
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            self.frame_static = img
            print(f"[INFO] Modo imagen estática: {image_path} -> {width}x{height}")
        else:
            self.cap = open_camera(index=cam_index, width=width, height=height, fps=int(1/self.fps_delay), prefer_mjpg=True)

    def read(self):
        if self.use_static:
            # tasa “FPS” aproximada para que no consuma 100% CPU
            now = time.time()
            if now - self._last_time < self.fps_delay:
                time.sleep(max(0.0, self.fps_delay - (now - self._last_time)))
            self._last_time = time.time()
            return True, self.frame_static.copy()
        else:
            return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()

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
    # Borde útil para referencia visual
    box = np.array([[[0,0]], [[1,0]], [[1,1]], [[0,1]], [[0,0]]], dtype=np.float32)
    box_t = cv2.perspectiveTransform(box, H).astype(int)
    cv2.polylines(img, [box_t.reshape(-1,2)], isClosed=True, color=(0,180,255), thickness=2)

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

    # Ejes X,Y de la malla
    ax = np.array([[[0,0]], [[1,0]]], dtype=np.float32)
    ay = np.array([[[0,0]], [[0,1]]], dtype=np.float32)
    ax_t = cv2.perspectiveTransform(ax, H).astype(int)
    ay_t = cv2.perspectiveTransform(ay, H).astype(int)
    cv2.arrowedLine(img, tuple(ax_t[0,0]), tuple(ax_t[1,0]), (255,80,80), 2, tipLength=0.08)
    cv2.arrowedLine(img, tuple(ay_t[0,0]), tuple(ay_t[1,0]), (80,80,255), 2, tipLength=0.08)
    cv2.putText(img, "X", tuple(ax_t[1,0]+[8,-8]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,80,80), 2)
    cv2.putText(img, "Y", tuple(ay_t[1,0]+[8,-8]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80,80,255), 2)

def pix_to_grid_uv(pt_pix, H_inv):
    """(x,y) imagen → (u,v) en [0,1]^2 via H_inv."""
    src = np.array([[[pt_pix[0], pt_pix[1]]]], dtype=np.float32)
    uv = cv2.perspectiveTransform(src, H_inv)[0,0]
    return float(uv[0]), float(uv[1])

# ======================================================================
#  Firebase + Control GOTO para enviar comandos al carrito
# ======================================================================
from firebase_admin import credentials as _cred, db as _db, initialize_app as _fb_init

# --- CONFIG --- (ajústalo si tu proyecto difiere)
CRED_PATH       = "cred.json"
DB_URL          = "https://iot-app-f878d-default-rtdb.firebaseio.com/"
ROBOT_ID_PATH   = "robots/123456"
INSTR_PATH      = f"{ROBOT_ID_PATH}/instrucciones"

# Ganancias: error (en celdas de la malla) -> comando (vx, vy). w según orientación.
Kp_lin          = 80.0     # velocidad por celda de error
Kp_w_face_move  = 120.0    # giro para orientar según dirección de avance
Kp_w_hold       = 120.0    # giro si se mantiene rumbo fijo

# Límites que espera tu carrito
CMD_MAX         = 250
W_MAX           = 250

# Tolerancias de llegada (pos y ángulo)
POS_TOL         = 0.20
ANG_TOL         = math.radians(6.0)

# Estados del controlador
_target_g = None           # (gx*, gy*)
_hold_heading_rad = None
orient_move = True         # True=orienta hacia avance; False=mantiene rumbo

# refs globales para callbacks
_instr_ref = None
CURRENT_H_INV = None       # homografía inversa accesible desde callback
_display_scale_ref = {"s": 0.75}  # para corregir clicks por zoom

# Inicializa Firebase
try:
    _fb_init(_cred.Certificate(CRED_PATH), {'databaseURL': DB_URL})
    _instr_ref = _db.reference(INSTR_PATH)
    print("[GOTO] Firebase conectado.")
except Exception as e:
    print(f"[GOTO][WARN] No se pudo inicializar Firebase: {e}")
    _instr_ref = None

def _send_cmd(vx, vy, w):
    """Publica en Firebase como cadenas, formato que tu carrito lee."""
    if _instr_ref is None:
        return
    try:
        _instr_ref.child("movimiento").update({"vx": str(int(vx)), "vy": str(int(vy))})
        _instr_ref.child("rotación").update({"w": str(int(w))})  # nota: clave con tilde
    except Exception as e:
        print(f"[GOTO][ERR] Publicando a Firebase: {e}")

def _stop_cmd():
    if _instr_ref is None:
        return
    try:
        _instr_ref.update({"parar": True})
        _instr_ref.child("movimiento").update({"vx": "0", "vy": "0"})
        _instr_ref.child("rotación").update({"w": "0"})
        print("[GOTO] Paro enviado.")
    except Exception as e:
        print(f"[GOTO][ERR] Paro: {e}")

def set_target(gx_star, gy_star, current_heading_deg=None):
    """Define objetivo en malla. Si orient_move=False, fija rumbo a mantener."""
    global _target_g, _hold_heading_rad
    _target_g = (float(gx_star), float(gy_star))
    if not orient_move and current_heading_deg is not None:
        _hold_heading_rad = math.radians(current_heading_deg)
    print(f"[GOTO] Nuevo objetivo: ({_target_g[0]:.2f}, {_target_g[1]:.2f})")

def clear_target():
    """Borra objetivo y detiene el carrito."""
    global _target_g, _hold_heading_rad
    _target_g = None
    _hold_heading_rad = None
    _stop_cmd()

def _sat(x, lim):
    return max(-lim, min(lim, x))

def goto_controller_step(gx, gy, heading_deg):
    """
    Llamar en cada frame cuando haya homografía y robot detectado.
    Entradas: gx, gy (malla), heading_deg (0° +X, CCW).
    Publica (vx, vy, w).
    """
    if _target_g is None:
        return

    ex = _target_g[0] - gx
    ey = _target_g[1] - gy
    dist = math.hypot(ex, ey)

    if dist <= POS_TOL:
        _send_cmd(0, 0, 0)
        # Mantiene posición; usa clear_target() con 'p' si quieres limpiar.
        print(f"[GOTO] Llegada: dist={dist:.2f}.")
        return

    # Control proporcional lineal
    vx_cmd = Kp_lin * ex
    vy_cmd = Kp_lin * ey

    # Limitar norma del vector
    norm = math.hypot(vx_cmd, vy_cmd)
    if norm > CMD_MAX:
        scale = CMD_MAX / (norm + 1e-9)
        vx_cmd *= scale
        vy_cmd *= scale

    th = math.radians(heading_deg)

    # Comando de giro
    if orient_move:
        if norm > 1e-3:
            th_des = math.atan2(vy_cmd, vx_cmd)
            e_th = math.atan2(math.sin(th_des - th), math.cos(th_des - th))
            w_cmd = Kp_w_face_move * e_th
        else:
            w_cmd = 0.0
    else:
        if _hold_heading_rad is None:
            _hold_heading_rad = th
        e_th = math.atan2(math.sin(_hold_heading_rad - th), math.cos(_hold_heading_rad - th))
        w_cmd = Kp_w_hold * e_th

    w_cmd = _sat(w_cmd, W_MAX)

    # Publicar
    _send_cmd(round(vx_cmd), round(vy_cmd), round(w_cmd))

def on_mouse(event, x, y, flags, param):
    """
    Click izquierdo: fija objetivo en la malla usando homografía actual.
    Corrige por escala de display.
    """
    global CURRENT_H_INV
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if CURRENT_H_INV is None:
        print("[GOTO] No hay homografía válida.")
        return

    scale = _display_scale_ref["s"]
    x_real = int(x / scale)
    y_real = int(y / scale)

    u, v_top = pix_to_grid_uv((x_real, y_real), CURRENT_H_INV)
    u = float(np.clip(u, 0.0, 1.0))
    v = 1.0 - float(np.clip(v_top, 0.0, 1.0))  # origen abajo-izq

    gx_star = u * GRID_N
    gy_star = v * GRID_N
    set_target(gx_star, gy_star)

# ==========================
#  Main
# ==========================
def main():
    global CURRENT_H_INV, orient_move  # para callback y tecla 'm'

    # Fuente única (cámara o imagen)
    src = FrameSource(
        use_static_image=USE_STATIC_IMAGE,
        image_path=IMAGE_PATH,
        width=TARGET_W, height=TARGET_H,
        fps=30, cam_index=1
    )

    cv2.namedWindow("Cuadricula ArUco 1080p", cv2.WINDOW_NORMAL)
    DISPLAY_SCALE = 0.75  # zoom de la ventana (solo visualización)
    _display_scale_ref["s"] = DISPLAY_SCALE
    cv2.setMouseCallback("Cuadricula ArUco 1080p", on_mouse)

    misses = 0
    MAX_MISSES = 30
    current_H_inv = None

    while True:
        ret, frame = src.read()
        if not ret or frame is None:
            misses += 1
            if not USE_STATIC_IMAGE and misses > MAX_MISSES:
                print("[WARN] Reintentando abrir cámara…")
                src.release()
                src = FrameSource(False, "", TARGET_W, TARGET_H, fps=30, cam_index=0)
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
                # ¡OJO! order_quad usa TL,TR,BR,BL según sus posiciones en imagen
                pts_dst = order_quad([centers_px[k] for k in centers_px.keys()])  # TL,TR,BR,BL
                pts_src = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]], dtype=np.float32)
                H = cv2.getPerspectiveTransform(pts_src, pts_dst)
                current_H_inv = np.linalg.inv(H)
                CURRENT_H_INV = current_H_inv  # expone a callback
                draw_grid_homography(frame, H, N=GRID_N, color=(0,255,0), thickness=1)
            else:
                current_H_inv = None
                CURRENT_H_INV = None

            # Robot en coords de malla (origen 0,0 abajo-izq)
            if (robot_id in ids) and (current_H_inv is not None):
                idx = list(ids).index(robot_id)
                c_robot = corners[idx][0]

                center_px = c_robot.mean(axis=0)
                px, py = int(center_px[0]), int(center_px[1])

                # (u,v) con origen arriba-izq -> invertimos v para abajo-izq
                u, v_top = pix_to_grid_uv((px, py), current_H_inv)
                u = float(np.clip(u, 0.0, 1.0))
                v = 1.0 - float(np.clip(v_top, 0.0, 1.0))

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

                # ====== Paso del controlador GOTO ======
                goto_controller_step(gx, gy, angle_grid)

        # --------- HUD de ayuda ----------
        hud = [
            "[Click izq] Fijar objetivo   [P] Parar   [M] Modo orientación",
            f"orient_move={'ON' if orient_move else 'OFF'}   Grid N={GRID_N}   +/- Zoom   ESC Salir"
        ]
        y0 = 28
        for i, line in enumerate(hud):
            cv2.putText(frame, line, (12, y0 + 22*i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20,20,20), 3)
            cv2.putText(frame, line, (12, y0 + 22*i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        # Overlay de modo imagen
        if USE_STATIC_IMAGE:
            txt = "MODO IMAGEN ESTATICA"
            cv2.putText(frame, txt, (12, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
            cv2.putText(frame, txt, (12, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # --------- Display escalado para que quepa en pantalla ---------
        disp = cv2.resize(frame, None, fx=_display_scale_ref["s"], fy=_display_scale_ref["s"],
                          interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Cuadricula ArUco 1080p", disp)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:    # ESC
            break
        elif k in (ord('+'), ord('=')):   # zoom in
            _display_scale_ref["s"] = min(1.0, _display_scale_ref["s"] + 0.05)
        elif k in (ord('-'), ord('_')):   # zoom out
            _display_scale_ref["s"] = max(0.30, _display_scale_ref["s"] - 0.05)
        elif k == ord('p'):               # parar y limpiar objetivo
            clear_target()
        elif k == ord('m'):               # alternar modo orientación
            orient_move = not orient_move
            print(f"[GOTO] orient_move = {orient_move}")

    src.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
