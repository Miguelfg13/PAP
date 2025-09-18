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
CRED_PATH       = "cred.json"
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
R  = 0.048      # Radio de rueda (m)
l1 = 0.097      # Distancia centro a rueda frontal (m)
l2 = 0.109      # Distancia centro a rueda trasera (m)

# Matriz cinemática inversa para ruedas mecanum
W  = (1 / R) * np.array([
    [ 1,  1, -(l1 + l2)],  # Rueda frontal izquierda
    [ 1, -1,  (l1 + l2)],  # Rueda frontal derecha  
    [ 1, -1, -(l1 + l2)],  # Rueda trasera izquierda
    [ 1,  1,  (l1 + l2)]   # Rueda trasera derecha
], dtype=float)

# Configuración de voltaje y PWM - COMPATIBLE CON INTERFAZ
VOLT_MAX = 9.0            # Voltaje máximo del motor
VOLT_MIN = 0.0            # Voltaje mínimo 
INPUT_MAX = 250           # Rango máximo de la interfaz (-250 a +250)
PWM_MAX = 255             # PWM máximo (0-255 para 8 bits)
DEADBAND_VOLT = 0.5       # Zona muerta en voltios
LOOP_DT = 0.05            # 20 Hz para mejor respuesta

# Polaridad física por rueda (ajusta según la instalación física)
MOTOR_POLARITY = np.array([1, -1, -1, 1], dtype=int)

# Factor de escala para convertir comandos de control a velocidades físicas
ESCALA_VEL = 50.0         # Escalado de comandos 0-9 a velocidades en mm/s

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

# Registro del robot
carritos_ref.child(id_final).set({
    "estado": "activo",
    "hostname": hostname,
    "timestamp": now_ts(),
    "volt_max": VOLT_MAX,
    "rango_input": f"±{INPUT_MAX}"
})
print(f"[INFO] Robot registrado como: {id_final}")
print(f"[INFO] Rango de comando: ±{INPUT_MAX} (equivale a ±{VOLT_MAX}V)")

# ==========================
#       I2C Setup
# ==========================
try:
    import smbus2 as smbus
    bus = smbus.SMBus(I2C_BUS_NUM)
    # Test de conectividad I2C
    try:
        bus.write_quick(DIRECCION_MOTORES)
        print(f"[INFO] Conexión I2C establecida en dirección {hex(DIRECCION_MOTORES)}")
    except Exception as e:
        print(f"[WARN] Test I2C falló: {e}")
    print("[INFO] Driver I2C inicializado.")
except Exception as e:
    print(f"[SIMULACIÓN] I2C no disponible: {e}")
    class FakeBus:
        def write_i2c_block_data(self, addr, reg, data):
            voltajes = [(d/255.0)*VOLT_MAX for d in data]
            print(f"[SIM] I2C -> Addr:{hex(addr)}, Reg:{hex(reg)}")
            print(f"      PWM: {data} -> Voltajes: [{voltajes[0]:.2f}V, {voltajes[1]:.2f}V, {voltajes[2]:.2f}V, {voltajes[3]:.2f}V]")
    bus = FakeBus()

# ==========================
#   Conversión y Control COMPATIBLE CON INTERFAZ
# ==========================
def comando_a_voltaje(cmd_value):
    """Convierte comando ±250 a voltaje ±9V"""
    # Saturar al rango de entrada
    cmd_value = max(-INPUT_MAX, min(INPUT_MAX, cmd_value))
    # Convertir proporcionalmente a voltaje
    return (cmd_value / INPUT_MAX) * VOLT_MAX

def voltaje_a_pwm(voltaje):
    """Convierte voltaje a PWM (0-255), manteniendo signo"""
    voltaje_abs = abs(voltaje)
    if voltaje_abs < DEADBAND_VOLT:
        return 0, 1  # PWM=0, dirección=adelante
    
    pwm = int((voltaje_abs / VOLT_MAX) * PWM_MAX)
    pwm = min(pwm, PWM_MAX)
    
    # Determinar dirección (1=adelante, -1=atrás)
    direccion = 1 if voltaje >= 0 else -1
    
    return pwm, direccion

def aplicar_deadband_comando(cmd):
    """Aplica zona muerta a nivel de comando"""
    deadband_cmd = INPUT_MAX * 0.02  # 2% del rango como zona muerta
    return 0 if abs(cmd) < deadband_cmd else cmd

def calcular_pwm_desde_comandos(cmd_vx, cmd_vy, cmd_w):
    """
    Convierte comandos ±250 a PWM para cada rueda
    
    Args:
        cmd_vx, cmd_vy, cmd_w: Comandos de entrada en rango ±250
        
    Returns:
        Lista de 4 valores PWM con dirección
    """
    # Aplicar zona muerta a comandos
    cmd_vx = aplicar_deadband_comando(cmd_vx)
    cmd_vy = aplicar_deadband_comando(cmd_vy) 
    cmd_w = aplicar_deadband_comando(cmd_w)
    
    # Los comandos ya vienen escalados desde la interfaz
    # Solo necesitamos aplicar la cinemática inversa
    vx = float(cmd_vx)  # Velocidad lineal X
    vy = float(cmd_vy)  # Velocidad lineal Y  
    w = float(cmd_w) * 0.5  # Velocidad angular (reducir para mejor control)
    
    # Calcular velocidades de rueda usando cinemática inversa simplificada
    # Para ruedas mecanum: [FL, FR, RL, RR]
    vel_ruedas = np.array([
        vx + vy - w,  # Frontal izquierda
        vx - vy + w,  # Frontal derecha
        vx - vy - w,  # Trasera izquierda  
        vx + vy + w   # Trasera derecha
    ], dtype=float)
    
    # Normalizar si alguna rueda excede el límite
    vel_max_actual = np.max(np.abs(vel_ruedas))
    if vel_max_actual > INPUT_MAX:
        factor_escala = INPUT_MAX / vel_max_actual
        vel_ruedas = vel_ruedas * factor_escala
        print(f"[INFO] Escalando velocidades por factor {factor_escala:.3f}")
    
    # Convertir a PWM y dirección para cada rueda
    pwm_values = []
    direcciones = []
    voltajes = []
    
    for i, vel in enumerate(vel_ruedas):
        # Aplicar polaridad de motor
        vel_con_polaridad = vel * MOTOR_POLARITY[i]
        
        # Convertir a voltaje
        voltaje = comando_a_voltaje(vel_con_polaridad)
        voltajes.append(voltaje)
        
        # Convertir a PWM con dirección
        pwm, direccion = voltaje_a_pwm(voltaje)
        pwm_values.append(pwm)
        direcciones.append(direccion)
    
    return pwm_values, direcciones, voltajes

def enviar_comandos_motor(cmd_vx, cmd_vy, cmd_w):
    """Envía comandos a los motores via I2C"""
    try:
        pwm_values, direcciones, voltajes = calcular_pwm_desde_comandos(cmd_vx, cmd_vy, cmd_w)
        
        # Para controladores que manejan dirección separada, 
        # aquí podrías enviar PWM y dirección por separado
        # Por ahora, enviamos PWM directamente
        bus.write_i2c_block_data(DIRECCION_MOTORES, REG_VELOCIDAD_FIJA, pwm_values)
        
        # Log para monitoreo (solo si hay movimiento)
        if any(cmd != 0 for cmd in [cmd_vx, cmd_vy, cmd_w]):
            print(f"[MOTOR] Cmd: vx={cmd_vx:>4} vy={cmd_vy:>4} w={cmd_w:>4}")
            print(f"        PWM: {pwm_values}")
            print(f"        Dir: {direcciones}")
            print(f"        V: [{voltajes[0]:.2f}, {voltajes[1]:.2f}, {voltajes[2]:.2f}, {voltajes[3]:.2f}]V")
            
    except Exception as e:
        print(f"[ERROR] Enviando a motores: {e}")

def detener_motores():
    """Para todos los motores inmediatamente"""
    try:
        bus.write_i2c_block_data(DIRECCION_MOTORES, REG_VELOCIDAD_FIJA, [0, 0, 0, 0])
        print("[INFO] Motores detenidos")
    except Exception as e:
        print(f"[WARN] Error deteniendo motores: {e}")

# ==========================
#  Terminación limpia
# ==========================
def cerrar_todo(signal_received=None, frame=None):
    """Cierre limpio del sistema"""
    print("\n[INFO] Iniciando cierre del sistema...")
    detener_motores()
    
    try:
        # Limpiar referencias de Firebase
        db.reference(f"{TERMINAR_PATH}/{id_final}").delete()
        heartbeat_ref.delete()
        carritos_ref.child(id_final).delete()
        print("[INFO] Referencias Firebase eliminadas")
    except Exception as e:
        print(f"[WARN] Error limpiando Firebase: {e}")
    
    print("[INFO] Sistema cerrado correctamente")
    sys.exit(0)

# Configurar manejadores de señales
signal.signal(signal.SIGINT,  cerrar_todo)
signal.signal(signal.SIGTERM, cerrar_todo)

# ==========================
#   Bucle principal optimizado
# ==========================
def escuchar_comandos():
    """Bucle principal de escucha y ejecución de comandos"""
    print(f"[INFO] Iniciando escucha de comandos...")
    print(f"[INFO] Ruta de instrucciones: {INSTR_PATH}")
    print(f"[INFO] Frecuencia de control: {1/LOOP_DT:.1f} Hz")
    
    last_hb = 0.0
    HB_DT = 2.0  # Heartbeat cada 2 segundos
    comando_anterior = {"vx": 0, "vy": 0, "w": 0}
    
    while True:
        t_inicio = time.time()

        # Heartbeat periódico
        if (t_inicio - last_hb) >= HB_DT:
            try:
                heartbeat_ref.set({
                    "por": id_final,
                    "hostname": hostname, 
                    "timestamp": now_ts(),
                    "estado": "activo"
                })
                last_hb = t_inicio
            except Exception as e:
                print(f"[WARN] Heartbeat falló: {e}")

        # Verificar señal de terminación
        try:
            if terminar_ref.get() is True:
                print("[INFO] Señal de terminación recibida")
                cerrar_todo()
        except Exception as e:
            print(f"[WARN] Error verificando terminación: {e}")

        # Leer instrucciones de Firebase
        try:
            data = instr_ref.get()
        except Exception as e:
            print(f"[WARN] Error leyendo instrucciones: {e}")
            time.sleep(LOOP_DT)
            continue

        # Parsear comandos
        cmd_vx = cmd_vy = cmd_w = 0
        pedir_paro = False

        if isinstance(data, dict):
            # Verificar comando de paro
            pedir_paro = bool(data.get("parar", False))
            
            # Extraer comandos de movimiento y rotación
            movimiento = data.get("movimiento", {}) or {}
            rotacion = data.get("rotación", data.get("rotacion", {})) or {}

            # Parsear valores - esperamos strings que representen números ±250
            try:
                vx_str = str(movimiento.get("vx", "0")).strip()
                vy_str = str(movimiento.get("vy", "0")).strip()  
                w_str = str(rotacion.get("w", "0")).strip()

                # Convertir y validar rango ±INPUT_MAX
                cmd_vx = max(-INPUT_MAX, min(INPUT_MAX, int(float(vx_str))))
                cmd_vy = max(-INPUT_MAX, min(INPUT_MAX, int(float(vy_str))))
                cmd_w = max(-INPUT_MAX, min(INPUT_MAX, int(float(w_str))))
                
            except (ValueError, TypeError) as e:
                print(f"[WARN] Comandos inválidos - vx:{vx_str} vy:{vy_str} w:{w_str} -> {e}")
                cmd_vx = cmd_vy = cmd_w = 0

        # Procesar comando de paro
        if pedir_paro:
            detener_motores()
            try:
                instr_ref.update({"parar": False})  # Limpiar bandera
            except Exception:
                pass
            comando_anterior = {"vx": 0, "vy": 0, "w": 0}
            time.sleep(LOOP_DT)
            continue

        # Ejecutar comando de movimiento
        comando_actual = {"vx": cmd_vx, "vy": cmd_vy, "w": cmd_w}
        
        # Solo enviar si hay cambio (reduce tráfico I2C)
        if comando_actual != comando_anterior:
            enviar_comandos_motor(cmd_vx, cmd_vy, cmd_w)
            comando_anterior = comando_actual.copy()
        
        # Mantener frecuencia de bucle constante
        dt_transcurrido = time.time() - t_inicio
        if dt_transcurrido < LOOP_DT:
            time.sleep(LOOP_DT - dt_transcurrido)

# ==========================
#         MAIN
# ==========================
def main():
    """Función principal"""
    print("="*60)
    print("    CONTROL ROBOT MECANUM - Compatible con Interfaz")
    print("="*60)
    print(f"Voltaje máximo: {VOLT_MAX}V")
    print(f"Rango comandos: ±{INPUT_MAX}")
    print(f"Frecuencia: {1/LOOP_DT:.1f} Hz")
    print("="*60)
    
    # Test inicial de motores
    print("[INFO] Test inicial de motores...")
    detener_motores()
    time.sleep(0.5)
    
    try:
        escuchar_comandos()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupción por teclado")
        cerrar_todo()
    except Exception as e:
        print(f"[ERROR] Error crítico: {e}")
        cerrar_todo()

if __name__ == "__main__":
    main()