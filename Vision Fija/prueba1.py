# === carrito_seguidor_instrucciones.py ===
# Lee la estructura EXACTA escrita por tu script de visión:
# robots/<ROBOT_ID>/instrucciones/movimiento/{vx, vy}  (strings)
# robots/<ROBOT_ID>/instrucciones/rotación/{w}        (string)
# robots/<ROBOT_ID>/instrucciones/parar = True        (flag)
#
# Requisitos: numpy, firebase_admin, (smbus2 si hay I2C real)

import os
import sys
import time
import uuid
import math
import socket
import signal
import argparse
import numpy as np

import firebase_admin
from firebase_admin import credentials, db

# ==========================
# Config Firebase (ajusta si cambias credenciales/URL)
# ==========================
CRED_PATH = "cred.json"
DB_URL= "https://iot-app-f878d-default-rtdb.firebaseio.com/"

# Nodos base (NO cambiar estructura de instrucciones)
ROOT_ROBOTS = "robots"

# ==========================
# Parámetros de motores / cinemática
# ==========================
R  = 0.048
l1 = 0.097
l2 = 0.109

Wm = (1.0 / R) * np.array([
    [1,  1, -(l1 + l2)],
    [1,  1,  (l1 + l2)],
    [1, -1,  (l1 + l2)],
    [1, -1, -(l1 + l2)]
], dtype=float)

V_MAX   = 250   # igual que tu normalización original
PWM_MAX = 100

# ==========================
# I2C
# ==========================
DIRECCION_MOTORES   = 0x34
REG_VELOCIDAD_FIJA  = 0x33

def abrir_i2c():
    try:
        import smbus2 as smbus
        bus = smbus.SMBus(1)
        # Algunos dispositivos no aceptan write_quick; no es crítico si falla
        try:
            bus.write_quick(DIRECCION_MOTORES)
        except Exception:
            pass
        print("[I2C] Bus OK")
        return bus, False
    except Exception as e:
        print(f"[I2C][Simulación] No disponible: {e}")
        class FakeBus:
            def write_i2c_block_data(self, addr, reg, data):
                print(f"[I2C Sim] addr={hex(addr)} reg={hex(reg)} data={data}")
        return FakeBus(), True

BUS, SIM_I2C = abrir_i2c()

def calcular_pwm(vx, vy, omega):
    V = np.array([vx, vy, omega], dtype=float)
    ruedas = Wm.dot(V)

    # Limitar magnitud como en tu código (a 250)
    maxabs = float(np.max(np.abs(ruedas))) if ruedas.size else 0.0
    if maxabs > 250:
        ruedas *= (250.0 / (maxabs + 1e-9))

    # Compensación de sentido para motores 2 y 3 (igual a tu script)
    ruedas[1] *= -1
    ruedas[2] *= -1

    pwm = np.clip((ruedas / V_MAX) * PWM_MAX, -PWM_MAX, PWM_MAX)
    return [int(round(p)) for p in pwm]

def enviar_pwm(vx, vy, omega):
    pwm = calcular_pwm(vx, vy, omega)
    BUS.write_i2c_block_data(DIRECCION_MOTORES, REG_VELOCIDAD_FIJA, pwm)

def detener_motores():
    try:
        BUS.write_i2c_block_data(DIRECCION_MOTORES, REG_VELOCIDAD_FIJA, [0, 0, 0, 0])
    except Exception as e:
        print(f"[WARN] Error al detener motores: {e}")

# ==========================
# Utilidades
# ==========================
def parse_args():
    p = argparse.ArgumentParser(description="Carrito seguidor de instrucciones (estructura de visión).")
    p.add_argument("--robot-id", type=str, default=os.environ.get("ROBOT_ID", "123456"),
                   help="ID del robot para ruta robots/<ROBOT_ID>/instrucciones (default: 123456 o env ROBOT_ID)")
    return p.parse_args()

def safe_int_str(s, default=0):
    try:
        if s is None: return default
        if isinstance(s, (int, float)): return int(round(s))
        return int(float(str(s)))
    except Exception:
        return default

# ==========================
# Limpieza
# ==========================
_shutdown = False

def cerrar_todo(_sig=None, _frm=None):
    global _shutdown
    if _shutdown: 
        sys.exit(0)
    _shutdown = True
    try:
        detener_motores()
    finally:
        sys.exit(0)

signal.signal(signal.SIGINT, cerrar_todo)
signal.signal(signal.SIGTERM, cerrar_todo)

# ==========================
# Main
# ==========================
def main():
    args = parse_args()
    ROBOT_ID = args.robot_id

    # Firebase init
    cred = credentials.Certificate(CRED_PATH)
    firebase_admin.initialize_app(cred, {'databaseURL': DB_URL})

    # Rutas
    instr_ref = db.reference(f"{ROOT_ROBOTS}/{ROBOT_ID}/instrucciones")
    estado_ref = db.reference(f"{ROOT_ROBOTS}/{ROBOT_ID}/estado")

    hostname = socket.gethostname()
    boot_id  = uuid.uuid4().hex[:6]
    print(f"[INIT] Carrito listo. ROBOT_ID={ROBOT_ID}  host={hostname}  boot={boot_id}  sim_i2c={SIM_I2C}")

    last_ping = 0
    last_cmd  = (None, None, None)  # (vx,vy,w) para evitar spamear el bus

    while not _shutdown:
        now = time.time()

        # Latido cada 1 s
        if now - last_ping >= 1.0:
            estado_ref.set({
                "hostname": hostname,
                "boot": boot_id,
                "sim_i2c": bool(SIM_I2C),
                "timestamp": now
            })
            last_ping = now

        data = instr_ref.get() or {}
        # Estructura esperada:
        # { "movimiento": {"vx": "12", "vy": "-30"}, "rotación": {"w": "15"}, "parar": True/False }
        parar_flag = bool(data.get("parar", False))

        if parar_flag:
            detener_motores()
            time.sleep(0.05)
            continue

        mov = data.get("movimiento", {}) or {}
        rot = data.get("rotación", {}) or {}

        vx = safe_int_str(mov.get("vx"), 0)
        vy = safe_int_str(mov.get("vy"), 0)
        w  = safe_int_str(rot.get("w"),  0)

        # Si no hay cambios, evita escribir al bus sin necesidad
        if (vx, vy, w) != last_cmd:
            try:
                if (vx, vy, w) == (0, 0, 0):
                    detener_motores()
                else:
                    enviar_pwm(vx, vy, w)
                # print(f"[CMD] vx={vx} vy={vy} w={w}")  # descomentar para debug
                last_cmd = (vx, vy, w)
            except Exception as e:
                print(f"[ERR] Envío I2C: {e}")
                detener_motores()

        time.sleep(0.05)  # ~20 Hz

if __name__ == "__main__":
    main()
