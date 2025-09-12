# vision_goto_aruco_calib.py
# Reqs: opencv-contrib-python, numpy, firebase_admin
# Ventanas: Click izq = fija objetivo; P=parar; M=modo orientación; C=calibrar; +/- zoom; ESC salir

import cv2
import numpy as np
import math
import time
import json, os

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

    if (aw, ah) != (width, height):
        if prefer_mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUY2'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS,          fps)
            aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            afps = cap.get(cv2.CAP_PROP_FPS)

    if (aw, ah) != (width, height):
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
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def draw_grid_homography(img, H, N=10, color=(0,255,0), thickness=1):
    box = np.array([[[0,0]], [[1,0]], [[1,1]], [[0,1]], [[0,0]]], dtype=np.float32)
    box_t = cv2.perspectiveTransform(box, H).astype(int)
    cv2.polylines(img, [box_t.reshape(-1,2)], True, (0,180,255), 2)

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

    ax = np.array([[[0,0]], [[1,0]]], dtype=np.float32)
    ay = np.array([[[0,0]], [[0,1]]], dtype=np.float32)
    ax_t = cv2.perspectiveTransform(ax, H).astype(int)
    ay_t = cv2.perspectiveTransform(ay, H).astype(int)
    cv2.arrowedLine(img, tuple(ax_t[0,0]), tuple(ax_t[1,0]), (255,80,80), 2, tipLength=0.08)
    cv2.arrowedLine(img, tuple(ay_t[0,0]), tuple(ay_t[1,0]), (80,80,255), 2, tipLength=0.08)
    cv2.putText(img, "X", tuple(ax_t[1,0]+[8,-8]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,80,80), 2)
    cv2.putText(img, "Y", tuple(ay_t[1,0]+[8,-8]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80,80,255), 2)

def pix_to_grid_uv(pt_pix, H_inv):
    src = np.array([[[pt_pix[0], pt_pix[1]]]], dtype=np.float32)
    uv = cv2.perspectiveTransform(src, H_inv)[0,0]
    return float(uv[0]), float(uv[1])

# ===== Dibujo de objetivo =====
CURRENT_H_INV = None
CURRENT_H     = None
_target_g     = None
_last_robot_px = None
_display_scale_ref = {"s": 0.75}

def grid_to_pix(gx, gy, H):
    if H is None:
        return None
    u = float(gx) / float(GRID_N)
    v_top = 1.0 - (float(gy) / float(GRID_N))
    pt = np.array([[[u, v_top]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(pt, H)[0,0]
    return int(dst[0]), int(dst[1])

def draw_target_marker(img, H):
    global _target_g
    if _target_g is None or H is None:
        return
    px = grid_to_pix(_target_g[0], _target_g[1], H)
    if px is None:
        return
    x, y = px
    cv2.circle(img, (x, y), 10, (0, 255, 255), 2)
    cv2.line(img, (x-14, y), (x+14, y), (0, 255, 255), 2)
    cv2.line(img, (x, y-14), (x, y+14), (0, 255, 255), 2)
    cv2.putText(img, f"Goal ({_target_g[0]:.2f},{_target_g[1]:.2f})", (x+10, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

# ======================================================================
#  Firebase + Control GOTO
# ======================================================================
from firebase_admin import credentials as _cred, db as _db, initialize_app as _fb_init

# --- CONFIG --- (ajústalo si tu proyecto difiere)
CRED_PATH       = "cred.json"
DB_URL          = "https://iot-app-f878d-default-rtdb.firebaseio.com/"
ROBOT_ID        = "123456"
ROBOT_ID_PATH   = f"robots/{ROBOT_ID}"
INSTR_PATH      = f"{ROBOT_ID_PATH}/instrucciones"

Kp_lin          = 80.0
Kp_w_face_move  = 120.0
Kp_w_hold       = 120.0
CMD_MAX         = 250
W_MAX           = 250
POS_TOL         = 0.20
ANG_TOL         = math.radians(6.0)

_target_g = None
_hold_heading_rad = None
orient_move = True

_instr_ref = None

try:
    _fb_init(_cred.Certificate(CRED_PATH), {'databaseURL': DB_URL})
    _instr_ref = _db.reference(INSTR_PATH)
    print("[GOTO] Firebase conectado.")
except Exception as e:
    print(f"[GOTO][WARN] No se pudo inicializar Firebase: {e}")
    _instr_ref = None

def _send_cmd(vx, vy, w):
    if _instr_ref is None:
        return
    try:
        _instr_ref.child("movimiento").update({"vx": str(int(vx)), "vy": str(int(vy))})
        _instr_ref.child("rotación").update({"w": str(int(w))})
        _instr_ref.update({"parar": False})
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
    global _target_g, _hold_heading_rad
    _target_g = (float(gx_star), float(gy_star))
    if not orient_move and current_heading_deg is not None:
        _hold_heading_rad = math.radians(current_heading_deg)
    print(f"[GOTO] Nuevo objetivo: ({_target_g[0]:.2f}, {_target_g[1]:.2f})")

def clear_target():
    global _target_g, _hold_heading_rad
    _target_g = None
    _hold_heading_rad = None
    _stop_cmd()

def _sat(x, lim):
    return max(-lim, min(lim, x))

# ===== Calibración =====
CALIB_PATH = "calib.json"
CAL = {"C": [[1.0, 0.0],[0.0, 1.0]], "k_w": 1.0}
_last_pose = {"gx": None, "gy": None, "ang_deg": None, "ts": 0.0}

def save_calib():
    try:
        with open(CALIB_PATH, "w", encoding="utf-8") as f:
            json.dump(CAL, f, ensure_ascii=False, indent=2)
        print(f"[CAL] Guardado en {CALIB_PATH}")
    except Exception as e:
        print(f"[CAL][ERR] Guardando: {e}")

def load_calib():
    global CAL
    if not os.path.exists(CALIB_PATH):
        return
    try:
        with open(CALIB_PATH, "r", encoding="utf-8") as f:
            CAL = json.load(f)
        print(f"[CAL] Cargado {CALIB_PATH}: C={CAL['C']}, k_w={CAL['k_w']:.4f}")
    except Exception as e:
        print(f"[CAL][WARN] No se pudo cargar {CALIB_PATH}: {e}")

def apply_calibration(vx_raw, vy_raw, w_raw):
    C = np.array(CAL["C"], dtype=float)
    k_w = float(CAL.get("k_w", 1.0))
    v_raw = np.array([vx_raw, vy_raw], dtype=float)
    v_cmd = C.dot(v_raw)
    w_cmd = (w_raw / k_w) if abs(k_w) > 1e-6 else w_raw
    return int(round(v_cmd[0])), int(round(v_cmd[1])), int(round(w_cmd))

def _send_cmd_calibrated(vx_raw, vy_raw, w_raw):
    vx, vy, w = apply_calibration(vx_raw, vy_raw, w_raw)
    _send_cmd(vx, vy, w)

def _update_last_pose(gx, gy, ang_deg):
    _last_pose.update({"gx": gx, "gy": gy, "ang_deg": ang_deg, "ts": time.time()})

def _wait_pose_fresh(timeout=2.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if _last_pose["gx"] is not None and (time.time() - _last_pose["ts"]) < 0.3:
            return True
        time.sleep(0.02)
    return False

def _impulse(cmd_vx, cmd_vy, cmd_w, T=1.2):
    if not _wait_pose_fresh():
        raise RuntimeError("No hay pose fresca para iniciar el impulso.")
    gx0, gy0 = _last_pose["gx"], _last_pose["gy"]
    a0 = math.radians(_last_pose["ang_deg"] or 0.0)

    _send_cmd(int(cmd_vx), int(cmd_vy), int(cmd_w))
    t0 = time.time()
    while time.time() - t0 < T:
        time.sleep(0.01)
    _stop_cmd()
    time.sleep(0.2)

    if not _wait_pose_fresh():
        raise RuntimeError("No hay pose fresca al finalizar el impulso.")
    gx1, gy1 = _last_pose["gx"], _last_pose["gy"]
    a1 = math.radians(_last_pose["ang_deg"] or 0.0)
    dth = math.atan2(math.sin(a1 - a0), math.cos(a1 - a0))
    return (gx1 - gx0, gy1 - gy0, dth, T)

def run_calibration():
    print("[CAL] Iniciando calibración… Asegúrate de ver el robot.")
    if not _wait_pose_fresh():
        print("[CAL][ERR] No hay pose del robot visible.")
        return
    U_lin = 120
    U_w   = 120
    T     = 1.2
    tests = [(+U_lin,0,0), (-U_lin,0,0), (0,+U_lin,0), (0,-U_lin,0)]
    obs=[]
    for (vx,vy,w) in tests:
        print(f"[CAL] Impulso lin: vx={vx}, vy={vy}")
        dgx, dgy, dth, Te = _impulse(vx,vy,w,T=T)
        obs.append((np.array([vx,vy],float), np.array([dgx/Te,dgy/Te],float)))
    Vcmd = np.stack([o[0] for o in obs], axis=0)
    Vobs = np.stack([o[1] for o in obs], axis=0)
    A_T  = np.linalg.pinv(Vcmd).dot(Vobs)
    A    = A_T.T
    print(f"[CAL] A estimada:\n{A}")
    print(f"[CAL] Impulso giro: w={U_w}")
    dgx, dgy, dth, Te = _impulse(0,0,U_w,T=T)
    omega_obs = dth/Te
    k_w = (omega_obs/float(U_w)) if (abs(U_w)>1e-6 and abs(omega_obs)>1e-6) else 1.0
    print(f"[CAL] k_w: {k_w:.6f}")
    try:
        C = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        C = np.linalg.pinv(A)
        print("[CAL][WARN] A no invertible, usando pseudoinversa.")
    global CAL
    CAL = {"C": C.tolist(), "k_w": float(k_w if abs(k_w)>1e-6 else 1.0)}
    save_calib()
    print("[CAL] ¡Listo! Se aplicará la corrección.")

# ===== Control GOTO =====
def goto_controller_step(gx, gy, heading_deg):
    if _target_g is None:
        return
    ex = _target_g[0] - gx
    ey = _target_g[1] - gy
    dist = math.hypot(ex, ey)
    if dist <= POS_TOL:
        _send_cmd_calibrated(0, 0, 0)
        print(f"[GOTO] Llegada: dist={dist:.2f}.")
        return
    vx_cmd = Kp_lin * ex
    vy_cmd = Kp_lin * ey
    norm = math.hypot(vx_cmd, vy_cmd)
    if norm > CMD_MAX:
        scale = CMD_MAX / (norm + 1e-9)
        vx_cmd *= scale
        vy_cmd *= scale
    th = math.radians(heading_deg)
    if orient_move:
        if norm > 1e-3:
            th_des = math.atan2(vy_cmd, vx_cmd)
            e_th = math.atan2(math.sin(th_des - th), math.cos(th_des - th))
            w_cmd = Kp_w_face_move * e_th
        else:
            w_cmd = 0.0
    else:
        global _hold_heading_rad
        if _hold_heading_rad is None:
            _hold_heading_rad = th
        e_th = math.atan2(math.sin(_hold_heading_rad - th), math.cos(_hold_heading_rad - th))
        w_cmd = Kp_w_hold * e_th
    w_cmd = _sat(w_cmd, W_MAX)
    _send_cmd_calibrated(round(vx_cmd), round(vy_cmd), round(w_cmd))

# ===== Mouse =====
def on_mouse(event, x, y, flags, param):
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
    v = 1.0 - float(np.clip(v_top, 0.0, 1.0))
    gx_star = u * GRID_N
    gy_star = v * GRID_N
    set_target(gx_star, gy_star)

# ==========================
#  Main
# ==========================
def main():
    global CURRENT_H_INV, CURRENT_H, orient_move, _last_robot_px
    load_calib()
    cap = open_camera(index=1, width=1920, height=1080, fps=30, prefer_mjpg=True)

    cv2.namedWindow("Cuadricula ArUco 1080p", cv2.WINDOW_NORMAL)
    DISPLAY_SCALE = 0.75
    _display_scale_ref["s"] = DISPLAY_SCALE
    cv2.setMouseCallback("Cuadricula ArUco 1080p", on_mouse)

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

        corners, ids, _ = detector.detectMarkers(frame)
        if ids is not None:
            ids = ids.flatten()
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, 0.05, camera_matrix, dist_coeffs)
            for i in range(len(ids)):
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03)

            centers_px = {}
            for i, mid in enumerate(ids):
                if mid in corner_ids:
                    c = corners[i][0]
                    centers_px[int(mid)] = c.mean(axis=0)

            if len(centers_px) == 4:
                pts_dst = order_quad([centers_px[k] for k in centers_px.keys()])
                pts_src = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]], dtype=np.float32)
                H = cv2.getPerspectiveTransform(pts_src, pts_dst)
                current_H_inv = np.linalg.inv(H)
                CURRENT_H = H
                CURRENT_H_INV = current_H_inv
                draw_grid_homography(frame, H, N=GRID_N, color=(0,255,0), thickness=1)
            else:
                current_H_inv = None
                CURRENT_H = None
                CURRENT_H_INV = None

            if (robot_id in ids) and (current_H_inv is not None):
                idx = list(ids).index(robot_id)
                c_robot = corners[idx][0]
                center_px = c_robot.mean(axis=0)
                px, py = int(center_px[0]), int(center_px[1])

                u, v_top = pix_to_grid_uv((px, py), current_H_inv)
                u = np.clip(u, 0.0, 1.0)
                v = 1.0 - np.clip(v_top, 0.0, 1.0)
                gx = u * GRID_N
                gy = v * GRID_N

                p_dir_img = (c_robot[0] + 0.5*(c_robot[1] - c_robot[0]))
                u0, v0_top = pix_to_grid_uv(center_px, current_H_inv)
                u1, v1_top = pix_to_grid_uv(p_dir_img, current_H_inv)
                v0 = 1.0 - v0_top
                v1 = 1.0 - v1_top
                du, dv = (u1 - u0), (v1 - v0)
                angle_grid = np.degrees(np.arctan2(dv, du))

                cv2.circle(frame, (px, py), 6, (0,0,255), -1)
                cv2.putText(frame, f"Robot ({gx:.2f}, {gy:.2f}) Ang: {angle_grid:.1f}°",
                            (px+10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                _last_robot_px = (px, py)
                _update_last_pose(gx, gy, angle_grid)
                goto_controller_step(gx, gy, angle_grid)

        draw_target_marker(frame, CURRENT_H)
        if _last_robot_px is not None and _target_g is not None and CURRENT_H is not None:
            tgt_px = grid_to_pix(_target_g[0], _target_g[1], CURRENT_H)
            if tgt_px is not None:
                cv2.line(frame, _last_robot_px, tgt_px, (0,200,255), 2)

        hud = [
            "[Click izq] Fijar objetivo   [P] Parar   [M] Modo orientación   [C] Calibrar",
            f"orient_move={'ON' if orient_move else 'OFF'}   Grid N={GRID_N}   +/- Zoom   ESC Salir"
        ]
        y0 = 28
        for i, line in enumerate(hud):
            cv2.putText(frame, line, (12, y0 + 22*i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20,20,20), 3)
            cv2.putText(frame, line, (12, y0 + 22*i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        disp = cv2.resize(frame, None, fx=_display_scale_ref["s"], fy=_display_scale_ref["s"], interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Cuadricula ArUco 1080p", disp)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k in (ord('+'), ord('=')):
            _display_scale_ref["s"] = min(1.0, _display_scale_ref["s"] + 0.05)
        elif k in (ord('-'), ord('_')):
            _display_scale_ref["s"] = max(0.30, _display_scale_ref["s"] - 0.05)
        elif k == ord('p'):
            clear_target()
        elif k == ord('m'):
            orient_move = not orient_move
            print(f"[GOTO] orient_move = {orient_move}")
        elif k == ord('c'):
            run_calibration()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
