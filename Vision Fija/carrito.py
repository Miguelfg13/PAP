import time
import uuid
import socket
import numpy as np
import firebase_admin
import signal
import sys
from firebase_admin import credentials, db

# ==========================
#           CONFIG
# ==========================
CRED_PATH       = "iot-app-f878d-firebase-adminsdk-fbsvc-9384c3ff98.json"
DB_URL          = "https://iot-app-f878d-default-rtdb.firebaseio.com/"

ROBOT_ID_PATH   = "robots/123456"               # <-- debe coincidir con el del script de visión
INSTR_PATH      = f"{ROBOT_ID_PATH}/instrucciones"
CARRITOS_PATH   = "carritos_disponibles"
HEARTBEAT_PATH  = "estado_conexion"
TERMINAR_PATH   = "terminar"

# I2C / Driver
DIRECCION_MOTORES   = 0x34
REG_VELOCIDAD_FIJA  = 0x33
I2C_BUS_NUM         = 1

# Cinemática y límites
R  = 0.048
l1 = 0.097
l2 = 0.109
W  = (1 / R) * np.array([
    [ 1,  1, -(l1 + l2)],
    [ 1,  1,  (l1 + l2)],
    [ 1, -1,  (l1 + l2)],
    [ 1, -1, -(l1 + l2)]
], dtype=float)

V_MAX = 250.0         # magnitud máx. velocidad rueda (antes de escalar)
PWM_MAX = 100         # rango de salida “lógico” por rueda (-100..100)
DEADBAND = 2          # zona muerta de PWM
LOOP_DT = 0.10        # s entre iteraciones (~10 Hz)

# Polaridad física por rueda (ajusta si alguna rueda gira al revés)
MOTOR_POLARITY = np.array([1, -1, -1, 1], dtype=int)

# Empaquetado de PWM al enviar por I2C:
# "SIGNED": -100..100 => 0..255 con offset 128 (habitual para bytes sin signo)
# "UNSIGNED": 0..100 directo; negativos => 0
PWM_PACKING = "SIGNED"     # cambia a "UNSIGNED" si tu driver lo requiere

# ==========================
#     Firebase Init
# ==========================
cred = credentials.Certificate(CRED_PATH)
firebase_admin.initialize_app(cred, {'databaseURL': DB_URL})

hostname = socket.gethostname()
id_final = f"carrito_{hostname}_{uuid.uuid4().hex[:6]}"
now_ts   = lambda: int(time.time())

carritos_ref  = db.reference(CARRITOS_PATH)
terminar_ref  = db.reference(f"{TERMINAR_PATH}/{id_final}")
instr_ref     = db.reference(INSTR_PATH)
heartbeat_ref = db.reference(f"{HEARTBEAT_PATH}/{id_final}")

carritos_ref.child(id_final).set({
    "estado": "activo",
    "hostname": hostname,
    "timestamp": now_ts()
})
print(f"[INFO] Carrito registrado como: {id_final}")

# ==========================
#       I2C Setup
# ==========================
try:
    import smbus2 as smbus
    bus = smbus.SMBus(I2C_BUS_NUM)
    # No todos soportan write_quick; si falla no es crítico
    try:
        bus.write_quick(DIRECCION_MOTORES)
    except Exception:
        pass
    print("[INFO] I2C real inicializado.")
except Exception as e:
    print(f"[Simulación] I2C no disponible: {e}")
    class FakeBus:
        def write_i2c_block_data(self, addr, reg, data):
            print(f"[Simulación] I2C -> Addr: {hex(addr)}, Reg: {hex(reg)}, Data: {data}")
    bus = FakeBus()

# ==========================
#   Utilidades PWM/I2C
# ==========================
def aplicar_deadband(p):
    return 0 if abs(p) < DEADBAND else p

def calcular_pwm(vx, vy, omega):
    """Mapea (vx,vy,w) -> 4 PWMs con saturación y polaridad."""
    V = np.array([vx, vy, omega], dtype=float)
    vel_ruedas = W @ V  # 4x1

    vmax = np.max(np.abs(vel_ruedas))
    if vmax > V_MAX:
        vel_ruedas = vel_ruedas * (V_MAX / vmax)

    pwm_float = (vel_ruedas / V_MAX) * PWM_MAX
    pwm_float = pwm_float * MOTOR_POLARITY

    pwm = [aplicar_deadband(int(np.clip(p, -PWM_MAX, PWM_MAX))) for p in pwm_float]
    return pwm  # signed (-100..100)

def pack_pwm(pwm_signed_list):
    """Empaqueta PWMs a bytes según PWM_PACKING."""
    if PWM_PACKING.upper() == "SIGNED":
        # -100..100 -> 0..255 (offset 128)
        out = []
        for p in pwm_signed_list:
            p = int(np.clip(p, -PWM_MAX, PWM_MAX))
            val = p + 128
            val = int(np.clip(val, 0, 255))
            out.append(val)
        return out
    elif PWM_PACKING.upper() == "UNSIGNED":
        out = []
        for p in pwm_signed_list:
            p = int(np.clip(p, 0, PWM_MAX))  # negativos a 0
            out.append(p)
        return out
    else:
        raise ValueError("PWM_PACKING inválido. Usa 'SIGNED' o 'UNSIGNED'.")

def enviar_pwm(vx, vy, omega):
    pwm_signed = calcular_pwm(vx, vy, omega)
    payload = pack_pwm(pwm_signed)
    bus.write_i2c_block_data(DIRECCION_MOTORES, REG_VELOCIDAD_FIJA, payload)

def detener_motores():
    try:
        if PWM_PACKING.upper() == "SIGNED":
            payload = pack_pwm([0, 0, 0, 0])
            bus.write_i2c_block_data(DIRECCION_MOTORES, REG_VELOCIDAD_FIJA, payload)
        else:
            bus.write_i2c_block_data(DIRECCION_MOTORES, REG_VELOCIDAD_FIJA, [0, 0, 0, 0])
    except Exception as e:
        print(f"[WARN] Error al detener motores: {e}")

# ==========================
#  Terminación limpia
# ==========================
def cerrar_todo(signal_received=None, frame=None):
    detener_motores()
    print("Motores apagados.")
    try:
        db.reference(f"{TERMINAR_PATH}/{id_final}").delete()
    except Exception as e:
        print(f"[WARN] No pude borrar terminar/: {e}")
    try:
        heartbeat_ref.delete()
    except Exception as e:
        print(f"[WARN] No pude borrar heartbeat: {e}")
    try:
        carritos_ref.child(id_final).delete()
        print("Datos en Firebase eliminados.")
    except Exception as e:
        print(f"[WARN] Error eliminando datos en Firebase: {e}")
    sys.exit(0)

signal.signal(signal.SIGINT,  cerrar_todo)
signal.signal(signal.SIGTERM, cerrar_todo)

# ==========================
#   Bucle principal
# ==========================
def escuchar_comandos():
    print("Escuchando comandos...")
    last_hb = 0.0
    HB_DT = 0.5

    while True:
        t0 = time.time()

        # Heartbeat (rate-limited)
        if (t0 - last_hb) >= HB_DT:
            try:
                heartbeat_ref.set({"por": id_final, "hostname": hostname, "timestamp": now_ts()})
                last_hb = t0
            except Exception as e:
                print(f"[WARN] Heartbeat fallo: {e}")

        # Señal de terminación externa
        try:
            if terminar_ref.get() is True:
                print("Se recibió señal de terminación desde el cliente.")
                cerrar_todo()
        except Exception as e:
            print(f"[WARN] Leyendo terminar/: {e}")

        # Leer instrucciones
        try:
            data = instr_ref.get()
        except Exception as e:
            print(f"[WARN] Leyendo instrucciones: {e}")
            time.sleep(LOOP_DT)
            continue

        # Estructura esperada:
        # { "movimiento": {"vx":"..","vy":".."}, "rotación":{"w":".."}, "parar": true|false }
        vx = vy = w = 0
        pedir_paro = False

        if isinstance(data, dict):
            pedir_paro = bool(data.get("parar", False))

            movimiento = data.get("movimiento", {}) or {}
            rotacion   = data.get("rotación", data.get("rotacion", {})) or {}

            vx_str = movimiento.get("vx", "0")
            vy_str = movimiento.get("vy", "0")
            w_str  = rotacion.get("w", "0")

            try:
                vx = int(float(vx_str))   # acepta "80" o "80.0"
                vy = int(float(vy_str))
                w  = int(float(w_str))
            except (ValueError, TypeError):
                print(f"[WARN] Valores inválidos vx={vx_str} vy={vy_str} w={w_str}; usando 0.")
                vx = vy = w = 0

        if pedir_paro:
            detener_motores()
            # Limpia la bandera 'parar' para no repetir
            try:
                instr_ref.update({"parar": False})
            except Exception:
                pass
            time.sleep(LOOP_DT)
            continue

        # Enviar a motores
        try:
            enviar_pwm(vx, vy, w)
            print(f"[CMD] vx={vx:>4} vy={vy:>4} w={w:>4}")
        except Exception as e:
            print(f"[ERROR] I2C al enviar PWM: {e}")

        # Respeta tiempo de ciclo
        dt = time.time() - t0
        if dt < LOOP_DT:
            time.sleep(LOOP_DT - dt)

# Main
try:
    escuchar_comandos()
except KeyboardInterrupt:
    print("Interrumpido.")
    cerrar_todo()
