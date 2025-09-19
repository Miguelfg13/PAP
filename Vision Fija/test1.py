# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math
import time
import threading
from collections import deque
import tkinter as tk
from tkinter import ttk, messagebox
from firebase_admin import credentials as _cred, db as _db, initialize_app as _fb_init
import json
import os
from PIL import Image, ImageTk
import queue

# ==========================
#  Clase principal de la aplicación
# ==========================
class ArucoRobotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Control Robot ArUco - Interfaz Optimizada")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')
        
        # Variables del sistema
        self.cap = None
        self.running = False
        self.camera_thread = None
        self.firebase_thread = None
        self.config_file = "robot_config.json"
        
        # Cola para comandos Firebase (evitar bloqueos)
        self.firebase_queue = queue.Queue(maxsize=10)
        
        # Variables para la visualización en tkinter
        self.video_label = None
        self.current_frame = None
        self.video_width = 640
        self.video_height = 480
        
        # Variables de configuración
        self.camera_index = tk.IntVar(value=1)
        self.camera_width = tk.IntVar(value=1920)
        self.camera_height = tk.IntVar(value=1080)
        self.camera_fps = tk.IntVar(value=30)
        self.robot_id = tk.IntVar(value=4)
        self.corner_ids = [tk.IntVar(value=1), tk.IntVar(value=2), tk.IntVar(value=5), tk.IntVar(value=3)]
        self.grid_n = tk.IntVar(value=10)
        self.camera_exposure = tk.IntVar(value=-6)
        self.camera_brightness = tk.DoubleVar(value=0.3)
        
        # Variables del control
        self.kp_lin = tk.DoubleVar(value=80.0)
        self.kp_w_face = tk.DoubleVar(value=120.0)
        self.kp_w_hold = tk.DoubleVar(value=120.0)
        self.cmd_max = tk.IntVar(value=250)
        self.w_max = tk.IntVar(value=250)
        self.pos_tolerance = tk.DoubleVar(value=0.20)
        
        # Variables de estado
        self.current_fps = tk.StringVar(value="0.0 Hz")
        self.robot_position = tk.StringVar(value="No detectado")
        self.target_position = tk.StringVar(value="Sin objetivo")
        self.connection_status = tk.StringVar(value="Desconectado")
        self.orient_mode = tk.BooleanVar(value=True)
        
        # Firebase
        self.firebase_url = tk.StringVar(value="https://iot-app-f878d-default-rtdb.firebaseio.com/")
        self.firebase_path = tk.StringVar(value="robots/123456")
        self.cred_path = tk.StringVar(value="cred.json")
        self.instr_ref = None
        self.firebase_connected = False
        
        # Variables del procesador de video
        self._target_g = None
        self._hold_heading_rad = None
        self.CURRENT_H = None
        self.CURRENT_H_INV = None
        self._last_robot_px = None
        self.display_scale = 0.75
        
        # Variables de optimización ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self._homography_cache = {"H": None, "H_inv": None, "timestamp": 0}
        
        # Variables de optimización de rendimiento
        self._last_command_time = 0
        self._command_interval = 0.1   # 10 Hz para comandos (más conservador)
        self._last_gui_update_time = 0
        self._gui_update_interval = 0.2  # 5 Hz para actualizaciones de GUI
        self._current_robot_heading = 0.0
        self._robot_detected = False
        self._frame_skip_counter = 0
        self._detection_skip = 1  # Procesar detección cada frame (más confiable)
        self._grid_redraw_counter = 0
        self._grid_redraw_interval = 5  # Redibujar malla cada 5 frames
        
        # Cache de conversiones pixel-grid
        self._conversion_cache = {}
        self._cache_timestamp = 0
        
        # Cargar configuración
        self.load_config()
        
        # Crear interfaz
        self.create_interface()
        
        # Configurar eventos de cierre
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def create_interface(self):
        
        """Crea la interfaz gráfica completa"""
        # Estilo
        style = ttk.Style()
        style.theme_use('alt')
        style.configure('Title.TLabel', font=('Arial', 12, 'bold'), background='#2b2b2b', foreground='white')
        style.configure('Header.TLabel', font=('Arial', 10, 'bold'), background='#2b2b2b', foreground='#4CAF50')
        style.configure('Status.TLabel', font=('Arial', 9), background='#2b2b2b', foreground='#FFC107')
        
        # Panel principal dividido
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel izquierdo - Controles
        left_frame = ttk.Frame(main_paned, relief=tk.RAISED, borderwidth=2)
        main_paned.add(left_frame, weight=1)
        
        # Panel derecho - Video y estado
        right_frame = ttk.Frame(main_paned, relief=tk.RAISED, borderwidth=2)
        main_paned.add(right_frame, weight=2)
        
        self.create_control_panel(left_frame)
        self.create_video_panel(right_frame)
        
    def create_control_panel(self, parent):
        """Crea el panel de controles"""
        # Título
        title_label = ttk.Label(parent, text="Control Robot ArUco", style='Title.TLabel')
        title_label.pack(pady=(10, 20))
        
        # Crear notebook para pestañas
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10)
        
        # Pestaña 1: Configuración del Robot
        robot_frame = ttk.Frame(notebook)
        notebook.add(robot_frame, text="Robot")
        self.create_robot_config(robot_frame)
        
        # Pestaña 2: Configuración de Cámara
        camera_frame = ttk.Frame(notebook)
        notebook.add(camera_frame, text="Camara")
        self.create_camera_config(camera_frame)
        
        # Pestaña 3: Control y Parámetros
        control_frame = ttk.Frame(notebook)
        notebook.add(control_frame, text="Control")
        self.create_control_config(control_frame)
        
        # Pestaña 4: Firebase
        firebase_frame = ttk.Frame(notebook)
        notebook.add(firebase_frame, text="Firebase")
        self.create_firebase_config(firebase_frame)

        # Pestaña 5: Control Manual
        manual_frame = ttk.Frame(notebook)
        notebook.add(manual_frame, text="Manual")
        self.create_manual_control(manual_frame)
        
        # Botones principales
        self.create_main_buttons(parent)
        
    def create_robot_config(self, parent):
        """Configuración del robot y marcadores ArUco"""
        # ID del Robot
        robot_group = ttk.LabelFrame(parent, text="Identificación del Robot", padding=10)
        robot_group.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(robot_group, text="ID del Robot ArUco:").grid(row=0, column=0, sticky=tk.W, pady=2)
        robot_spin = ttk.Spinbox(robot_group, from_=0, to=50, textvariable=self.robot_id, width=10)
        robot_spin.grid(row=0, column=1, padx=(10,0), pady=2)
        
        # IDs de las esquinas
        corner_group = ttk.LabelFrame(parent, text="IDs de Esquinas de la Mesa", padding=10)
        corner_group.pack(fill=tk.X, padx=10, pady=5)
        
        corner_labels = ["Esquina 1:", "Esquina 2:", "Esquina 3:", "Esquina 4:"]
        for i, label_text in enumerate(corner_labels):
            ttk.Label(corner_group, text=label_text).grid(row=i//2, column=(i%2)*2, sticky=tk.W, pady=2, padx=(0,5))
            spin = ttk.Spinbox(corner_group, from_=0, to=50, textvariable=self.corner_ids[i], width=8)
            spin.grid(row=i//2, column=(i%2)*2+1, padx=(5,10), pady=2)
        
        # Configuración de la malla
        grid_group = ttk.LabelFrame(parent, text="Configuración de Malla", padding=10)
        grid_group.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(grid_group, text="Tamaño de Malla (NxN):").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(grid_group, from_=5, to=20, textvariable=self.grid_n, width=10).grid(row=0, column=1, padx=(10,0), pady=2)
        
    def create_camera_config(self, parent):
        """Configuración de la cámara"""
        # Configuración básica
        basic_group = ttk.LabelFrame(parent, text="Configuración Básica", padding=10)
        basic_group.pack(fill=tk.X, padx=10, pady=5)
        
        configs = [
            ("Índice de Cámara:", self.camera_index, 0, 5),
            ("Ancho (píxeles):", self.camera_width, 640, 1920),
            ("Alto (píxeles):", self.camera_height, 480, 1080),
            ("FPS Objetivo:", self.camera_fps, 15, 60)
        ]
        
        for i, (label_text, var, min_val, max_val) in enumerate(configs):
            ttk.Label(basic_group, text=label_text).grid(row=i, column=0, sticky=tk.W, pady=2)
            ttk.Spinbox(basic_group, from_=min_val, to=max_val, textvariable=var, width=10).grid(row=i, column=1, padx=(10,0), pady=2)
            ttk.Label(basic_group, text="Exposición (-11 a -1):").grid(row=4, column=0, sticky=tk.W, pady=2)
            ttk.Spinbox(basic_group, from_=-11, to=-1, textvariable=self.camera_exposure, width=10).grid(row=4, column=1, padx=(10,0), pady=2)

            ttk.Label(basic_group, text="Brillo (0.0 a 1.0):").grid(row=5, column=0, sticky=tk.W, pady=2)
            ttk.Scale(basic_group, from_=0.0, to=1.0, variable=self.camera_brightness, orient=tk.HORIZONTAL, length=100).grid(row=5, column=1, padx=(10,0), pady=2)
        
        # Resoluciones predefinidas
        res_group = ttk.LabelFrame(parent, text="Resoluciones Rápidas", padding=10)
        res_group.pack(fill=tk.X, padx=10, pady=5)
        
        res_buttons = [
            ("1080p", 1920, 1080),
            ("720p", 1280, 720),
            ("480p", 640, 480)
        ]
        
        for i, (name, w, h) in enumerate(res_buttons):
            btn = ttk.Button(res_group, text=name, 
                           command=lambda w=w, h=h: self.set_resolution(w, h))
            btn.grid(row=0, column=i, padx=5, pady=5)
            
    def create_control_config(self, parent):
        """Configuración de parámetros de control"""
        # Ganancias del controlador
        gains_group = ttk.LabelFrame(parent, text="Ganancias del Controlador", padding=10)
        gains_group.pack(fill=tk.X, padx=10, pady=5)
        
        gain_configs = [
            ("Kp Lineal:", self.kp_lin, 1.0, 200.0, 1.0),
            ("Kp Angular (Movimiento):", self.kp_w_face, 1.0, 300.0, 1.0),
            ("Kp Angular (Fijo):", self.kp_w_hold, 1.0, 300.0, 1.0)
        ]
        
        for i, (label_text, var, min_val, max_val, increment) in enumerate(gain_configs):
            ttk.Label(gains_group, text=label_text).grid(row=i, column=0, sticky=tk.W, pady=2)
            scale = ttk.Scale(gains_group, from_=min_val, to=max_val, variable=var, 
                            orient=tk.HORIZONTAL, length=150)
            scale.grid(row=i, column=1, padx=(10,5), pady=2)
            value_label = ttk.Label(gains_group, text=f"{var.get():.1f}")
            value_label.grid(row=i, column=2, pady=2)
            
            # Actualizar etiqueta cuando cambie el valor
            var.trace_add('write', lambda *args, lbl=value_label, v=var: lbl.configure(text=f"{v.get():.1f}"))
        
        # Límites y tolerancias
        limits_group = ttk.LabelFrame(parent, text="Límites y Tolerancias", padding=10)
        limits_group.pack(fill=tk.X, padx=10, pady=5)
        
        limit_configs = [
            ("Vel. Max Lineal:", self.cmd_max, 50, 500),
            ("Vel. Max Angular:", self.w_max, 50, 500),
            ("Tolerancia Posición:", self.pos_tolerance, 0.1, 1.0, 0.05)
        ]
        
        for i, config in enumerate(limit_configs):
            label_text, var = config[:2]
            ttk.Label(limits_group, text=label_text).grid(row=i, column=0, sticky=tk.W, pady=2)
            if len(config) > 4:  # Es float
                min_val, max_val, increment = config[2:]
                scale = ttk.Scale(limits_group, from_=min_val, to=max_val, variable=var, 
                                orient=tk.HORIZONTAL, length=100)
                scale.grid(row=i, column=1, padx=(10,5), pady=2)
                value_label = ttk.Label(limits_group, text=f"{var.get():.2f}")
                value_label.grid(row=i, column=2, pady=2)
                var.trace_add('write', lambda *args, lbl=value_label, v=var: lbl.configure(text=f"{v.get():.2f}"))
            else:  # Es int
                min_val, max_val = config[2:]
                ttk.Spinbox(limits_group, from_=min_val, to=max_val, textvariable=var, width=10).grid(row=i, column=1, padx=(10,0), pady=2)
        
    def create_firebase_config(self, parent):
        """Configuración de Firebase"""
        # Configuración de conexión
        conn_group = ttk.LabelFrame(parent, text="Configuración de Conexión", padding=10)
        conn_group.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(conn_group, text="URL de Firebase:").grid(row=0, column=0, sticky=tk.W, pady=2)
        url_entry = ttk.Entry(conn_group, textvariable=self.firebase_url, width=30)
        url_entry.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=2)
        
        ttk.Label(conn_group, text="Ruta del Robot:").grid(row=2, column=0, sticky=tk.W, pady=2)
        path_entry = ttk.Entry(conn_group, textvariable=self.firebase_path, width=30)
        path_entry.grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=2)
        
        ttk.Label(conn_group, text="Archivo de Credenciales:").grid(row=4, column=0, sticky=tk.W, pady=2)
        cred_frame = ttk.Frame(conn_group)
        cred_frame.grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=2)
        
        cred_entry = ttk.Entry(cred_frame, textvariable=self.cred_path)
        cred_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        browse_btn = ttk.Button(cred_frame, text="Examinar", width=10, command=self.browse_credentials)
        browse_btn.pack(side=tk.RIGHT, padx=(5,0))
        
        # Estado de conexión
        status_group = ttk.LabelFrame(parent, text="Estado de Conexión", padding=10)
        status_group.pack(fill=tk.X, padx=10, pady=5)
        
        status_label = ttk.Label(status_group, textvariable=self.connection_status, style='Status.TLabel')
        status_label.pack(pady=5)
        
        # Botones de conexión
        btn_frame = ttk.Frame(status_group)
        btn_frame.pack(fill=tk.X, pady=5)
        
        connect_btn = ttk.Button(btn_frame, text="Conectar", command=self.connect_firebase)
        connect_btn.pack(side=tk.LEFT, padx=(0,5))
        
        disconnect_btn = ttk.Button(btn_frame, text="Desconectar", command=self.disconnect_firebase)
        disconnect_btn.pack(side=tk.LEFT)

    def create_manual_control(self, parent):
        """Control manual del robot"""
        # Título
        title_group = ttk.LabelFrame(parent, text="Control Manual del Robot", padding=10)
        title_group.pack(fill=tk.X, padx=10, pady=5)
        
        # Marco para botones de dirección
        direction_frame = ttk.Frame(title_group)
        direction_frame.pack(pady=10)
        
        # Velocidad manual
        speed_frame = ttk.Frame(title_group)
        speed_frame.pack(pady=5)
        ttk.Label(speed_frame, text="Velocidad:").pack(side=tk.LEFT)
        self.manual_speed = tk.IntVar(value=100)
        ttk.Scale(speed_frame, from_=50, to=250, variable=self.manual_speed, 
                orient=tk.HORIZONTAL, length=150).pack(side=tk.LEFT, padx=10)
        speed_label = ttk.Label(speed_frame, text="100")
        speed_label.pack(side=tk.LEFT)
        self.manual_speed.trace_add('write', lambda *args: speed_label.configure(text=str(self.manual_speed.get())))
        
        # Disposición de botones en cruz
        #     ↑
        #   ← ● →
        #     ↓
        
        # Botón Adelante
        btn_up = ttk.Button(direction_frame, text="↑", width=3, command=lambda: self.manual_command(0, self.manual_speed.get(), 0))
        btn_up.grid(row=0, column=1, padx=2, pady=2)
        
        # Botones Izquierda y Derecha
        btn_left = ttk.Button(direction_frame, text="←", width=3, command=lambda: self.manual_command(-self.manual_speed.get(), 0, 0))
        btn_left.grid(row=1, column=0, padx=2, pady=2)
        
        btn_right = ttk.Button(direction_frame, text="→", width=3, command=lambda: self.manual_command(self.manual_speed.get(), 0, 0))
        btn_right.grid(row=1, column=2, padx=2, pady=2)
        
        # Botón Atrás
        btn_down = ttk.Button(direction_frame, text="↓", width=3, command=lambda: self.manual_command(0, -self.manual_speed.get(), 0))
        btn_down.grid(row=2, column=1, padx=2, pady=2)
        
        # Botones de rotación
        rotate_frame = ttk.Frame(title_group)
        rotate_frame.pack(pady=10)
        
        ttk.Button(rotate_frame, text="↻ Girar Izq", command=lambda: self.manual_command(0, 0, -self.manual_speed.get())).pack(side=tk.LEFT, padx=5)
        ttk.Button(rotate_frame, text="↺ Girar Der", command=lambda: self.manual_command(0, 0, self.manual_speed.get())).pack(side=tk.LEFT, padx=5)
        
        # Botón de parada
        ttk.Button(title_group, text="⏹ PARAR", command=lambda: self.manual_command(0, 0, 0)).pack(pady=10)
        
    def create_main_buttons(self, parent):
        """Botones principales de control"""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Botón de iniciar/parar
        self.start_btn = ttk.Button(button_frame, text="INICIAR SISTEMA", 
                                   command=self.toggle_system, style='Accent.TButton')
        self.start_btn.pack(fill=tk.X, pady=5)
        
        # Botones secundarios
        btn_row1 = ttk.Frame(button_frame)
        btn_row1.pack(fill=tk.X, pady=2)
        
        save_btn = ttk.Button(btn_row1, text="Guardar Config", command=self.save_config)
        save_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,2))
        
        load_btn = ttk.Button(btn_row1, text="Cargar Config", command=self.load_config)
        load_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2,0))
        
        btn_row2 = ttk.Frame(button_frame)
        btn_row2.pack(fill=tk.X, pady=2)
        
        stop_robot_btn = ttk.Button(btn_row2, text="Parar Robot", command=self.stop_robot)
        stop_robot_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,2))
        
        clear_target_btn = ttk.Button(btn_row2, text="Limpiar Objetivo", command=self.clear_target)
        clear_target_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2,0))
        
        # Modo de orientación
        orient_frame = ttk.Frame(button_frame)
        orient_frame.pack(fill=tk.X, pady=5)
        
        orient_check = ttk.Checkbutton(orient_frame, text="Orientar hacia movimiento", 
                                     variable=self.orient_mode)
        orient_check.pack()
        
    def create_video_panel(self, parent):
        """Panel de video y estado del sistema"""
        # Título del panel
        video_title = ttk.Label(parent, text="Vista de Camara y Estado del Sistema", style='Title.TLabel')
        video_title.pack(pady=(10, 5))
        
        # Frame para video con scroll si es necesario
        video_container = ttk.Frame(parent, relief=tk.SUNKEN, borderwidth=2)
        video_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Label para mostrar el video
        self.video_label = ttk.Label(video_container, 
                                   text="Vista de camara aparecera aqui\ncuando se inicie el sistema\n\nClick para establecer objetivo", 
                                   font=('Arial', 12), 
                                   anchor=tk.CENTER,
                                   foreground='gray',
                                   background='black')
        self.video_label.pack(expand=True, fill=tk.BOTH)
        
        # Bind del click del mouse
        self.video_label.bind("<Button-1>", self.on_video_click)
        
        # Panel de estado
        status_frame = ttk.LabelFrame(parent, text="Estado del Sistema", padding=10)
        status_frame.pack(fill=tk.X, padx=10, pady=(0,10))
        
        # Crear grid de estado
        status_items = [
            ("Frecuencia de Camara:", self.current_fps),
            ("Posición del Robot:", self.robot_position),
            ("Objetivo Actual:", self.target_position),
            ("Estado Firebase:", self.connection_status)
        ]
        
        for i, (label_text, var) in enumerate(status_items):
            ttk.Label(status_frame, text=label_text).grid(row=i//2, column=(i%2)*2, sticky=tk.W, padx=5, pady=2)
            status_label = ttk.Label(status_frame, textvariable=var, style='Status.TLabel')
            status_label.grid(row=i//2, column=(i%2)*2+1, sticky=tk.W, padx=5, pady=2)
    
    def on_video_click(self, event):
        """Manejar click en el video - OPTIMIZADO"""
        if self.CURRENT_H_INV is None or not self.running:
            return
        
        # Obtener coordenadas relativas al widget
        widget_x = event.x
        widget_y = event.y
        
        # Calcular la escala y offset del video en el widget
        widget_width = self.video_label.winfo_width()
        widget_height = self.video_label.winfo_height()
        
        if self.current_frame is not None:
            frame_height, frame_width = self.current_frame.shape[:2]
            
            # Calcular escala manteniendo aspecto
            scale_x = widget_width / frame_width
            scale_y = widget_height / frame_height
            scale = min(scale_x, scale_y)
            
            # Calcular offset para centrar
            scaled_width = frame_width * scale
            scaled_height = frame_height * scale
            offset_x = (widget_width - scaled_width) / 2
            offset_y = (widget_height - scaled_height) / 2
            
            # Convertir coordenadas del widget a coordenadas del frame original
            if (widget_x >= offset_x and widget_x <= offset_x + scaled_width and
                widget_y >= offset_y and widget_y <= offset_y + scaled_height):
                
                frame_x = (widget_x - offset_x) / scale
                frame_y = (widget_y - offset_y) / scale
                
                # Convertir píxeles a coordenadas de la malla
                u, v_top = self.pix_to_grid_uv_fast((frame_x, frame_y), self.CURRENT_H_INV)
                u = np.clip(u, 0.0, 1.0)
                v = 1.0 - np.clip(v_top, 0.0, 1.0)
                
                gx = u * self.grid_n.get()
                gy = v * self.grid_n.get()
                
                # Establecer objetivo
                current_heading = self._current_robot_heading if self._robot_detected else None
                self.set_target(gx, gy, current_heading)
                print(f"[CLICK] Nuevo objetivo en ({gx:.2f}, {gy:.2f})")
        
    # ==========================
    #  Métodos de configuración
    # ==========================
    def set_resolution(self, width, height):
        """Establece resolución predefinida"""
        self.camera_width.set(width)
        self.camera_height.set(height)
        
    def browse_credentials(self):
        """Examinar archivo de credenciales"""
        from tkinter import filedialog
        filename = filedialog.askopenfilename(
            title="Seleccionar archivo de credenciales",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.cred_path.set(filename)
            
    def save_config(self):
        """Guardar configuración actual"""
        config = {
            'camera': {
                'index': self.camera_index.get(),
                'width': self.camera_width.get(),
                'height': self.camera_height.get(),
                'fps': self.camera_fps.get()
            },
            'robot': {
                'id': self.robot_id.get(),
                'corner_ids': [var.get() for var in self.corner_ids],
                'grid_n': self.grid_n.get()
            },
            'control': {
                'kp_lin': self.kp_lin.get(),
                'kp_w_face': self.kp_w_face.get(),
                'kp_w_hold': self.kp_w_hold.get(),
                'cmd_max': self.cmd_max.get(),
                'w_max': self.w_max.get(),
                'pos_tolerance': self.pos_tolerance.get()
            },
            'firebase': {
                'url': self.firebase_url.get(),
                'path': self.firebase_path.get(),
                'cred_path': self.cred_path.get()
            }
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
            messagebox.showinfo("Éxito", f"Configuración guardada en {self.config_file}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar la configuración:\n{e}")
            
    def load_config(self):
        """Cargar configuración desde archivo"""
        if not os.path.exists(self.config_file):
            return
            
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                
            # Cargar configuración de cámara
            if 'camera' in config:
                cam = config['camera']
                self.camera_index.set(cam.get('index', 1))
                self.camera_width.set(cam.get('width', 1920))
                self.camera_height.set(cam.get('height', 1080))
                self.camera_fps.set(cam.get('fps', 30))
                
            # Cargar configuración de robot
            if 'robot' in config:
                robot = config['robot']
                self.robot_id.set(robot.get('id', 4))
                corner_ids = robot.get('corner_ids', [1, 2, 5, 3])
                for i, corner_id in enumerate(corner_ids[:4]):
                    self.corner_ids[i].set(corner_id)
                self.grid_n.set(robot.get('grid_n', 10))
                
            # Cargar configuración de control
            if 'control' in config:
                ctrl = config['control']
                self.kp_lin.set(ctrl.get('kp_lin', 80.0))
                self.kp_w_face.set(ctrl.get('kp_w_face', 120.0))
                self.kp_w_hold.set(ctrl.get('kp_w_hold', 120.0))
                self.cmd_max.set(ctrl.get('cmd_max', 250))
                self.w_max.set(ctrl.get('w_max', 250))
                self.pos_tolerance.set(ctrl.get('pos_tolerance', 0.20))
                
            # Cargar configuración de Firebase
            if 'firebase' in config:
                fb = config['firebase']
                self.firebase_url.set(fb.get('url', ''))
                self.firebase_path.set(fb.get('path', ''))
                self.cred_path.set(fb.get('cred_path', ''))
                
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la configuración:\n{e}")
            
    # ==========================
    #  Métodos de Firebase OPTIMIZADOS
    # ==========================
    def connect_firebase(self):
        """Conectar a Firebase"""
        try:
            if not os.path.exists(self.cred_path.get()):
                messagebox.showerror("Error", "Archivo de credenciales no encontrado")
                return
                
            _fb_init(_cred.Certificate(self.cred_path.get()), 
                    {'databaseURL': self.firebase_url.get()})
            
            instr_path = f"{self.firebase_path.get()}/instrucciones"
            self.instr_ref = _db.reference(instr_path)
            self.firebase_connected = True
            
            # Iniciar hilo de Firebase
            self.firebase_thread = threading.Thread(target=self.firebase_worker, daemon=True)
            self.firebase_thread.start()
            
            self.connection_status.set("Conectado")
            messagebox.showinfo("Éxito", "Conectado a Firebase correctamente")
            
        except Exception as e:
            self.connection_status.set("Error de conexión")
            messagebox.showerror("Error Firebase", f"No se pudo conectar a Firebase:\n{e}")
    
    def firebase_worker(self):
        """Hilo separado para procesar comandos de Firebase - EVITA BLOQUEOS"""
        while self.running and self.firebase_connected:
            try:
                # Procesar comandos de la cola con timeout
                try:
                    cmd_data = self.firebase_queue.get(timeout=0.1)
                    if cmd_data is None:  # Señal de parada
                        break
                    
                    # Enviar comando a Firebase
                    if self.instr_ref:
                        if cmd_data['type'] == 'movement':
                            self.instr_ref.child("movimiento").update({
                                "vx": str(cmd_data['vx']), 
                                "vy": str(cmd_data['vy'])
                            })
                        elif cmd_data['type'] == 'rotation':
                            self.instr_ref.child("rotación").update({
                                "w": str(cmd_data['w'])
                            })
                        elif cmd_data['type'] == 'stop':
                            self.instr_ref.update({"parar": True})
                            self.instr_ref.child("movimiento").update({"vx": "0", "vy": "0"})
                            self.instr_ref.child("rotación").update({"w": "0"})
                    
                    self.firebase_queue.task_done()
                    
                except queue.Empty:
                    continue
                    
            except Exception as e:
                print(f"[FIREBASE][ERR] Error en worker: {e}")
                time.sleep(0.1)
    
    def disconnect_firebase(self):
        """Desconectar Firebase"""
        self.firebase_connected = False
        if self.firebase_thread:
            # Enviar señal de parada al hilo
            try:
                self.firebase_queue.put_nowait(None)
            except queue.Full:
                pass
        self.instr_ref = None
        self.connection_status.set("Desconectado")
        
    # ==========================
    #  Métodos del sistema principal OPTIMIZADOS
    # ==========================
    def toggle_system(self):
        """Iniciar/parar el sistema"""
        if not self.running:
            self.start_system()
        else:
            self.stop_system()
            
    def start_system(self):
        """Iniciar el sistema de detección y control"""
        try:
            # Abrir cámara
            self.cap = self.open_camera()
            
            # Cambiar botón
            self.start_btn.configure(text="PARAR SISTEMA")
            
            # Iniciar hilo de cámara
            self.running = True
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
            messagebox.showinfo("Sistema Iniciado", "Sistema iniciado correctamente")
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo iniciar el sistema:\n{e}")
            
    def stop_system(self):
        """Parar el sistema"""
        self.running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
            
        # Limpiar el video label
        self.video_label.configure(image='', text="Vista de camara aparecera aqui\ncuando se inicie el sistema\n\nClick para establecer objetivo")
        self.current_frame = None
        
        self.start_btn.configure(text="INICIAR SISTEMA")
        self.current_fps.set("0.0 Hz")
        self.robot_position.set("No detectado")
        
        messagebox.showinfo("Sistema Detenido", "Sistema detenido correctamente")
        
    def open_camera(self):
        """Abrir cámara con configuración actual"""
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
        
        cap = None
        for backend in backends:
            cap = cv2.VideoCapture(self.camera_index.get(), backend)
            if cap.isOpened():
                break
            cap.release()
                
        if not cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara")
            
        # Configurar cámara
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, self.camera_fps.get())
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width.get())
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height.get())
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        # Control de exposición y brillo
        cap.set(cv2.CAP_PROP_EXPOSURE, self.camera_exposure.get())
        cap.set(cv2.CAP_PROP_BRIGHTNESS, self.camera_brightness.get())
        
        return cap
        
    def camera_loop(self):
        """Loop principal de la cámara - BALANCEADO para detección confiable"""
        fps_history = deque(maxlen=20)
        fps_counter = 0
        current_fps = 0.0
        misses = 0
        MAX_MISSES = 10
        
        # Variables para procesamiento optimizado
        last_fps_update = 0
        
        while self.running:
            loop_start = time.time()
            
            ret, frame = self.cap.read()
            if not ret or frame is None:
                misses += 1
                if misses > MAX_MISSES:
                    print("[WARN] Reintentando abrir cámara...")
                    self.cap.release()
                    time.sleep(0.1)
                    try:
                        self.cap = self.open_camera()
                    except:
                        self.root.after(0, lambda: messagebox.showerror("Error", "Se perdió la conexión con la cámara"))
                        break
                    misses = 0
                continue
            misses = 0
            
            # Calcular FPS real solo cada cierto tiempo
            fps_counter += 1
            fps_history.append(loop_start)
            
            if loop_start - last_fps_update > 0.5:  # Actualizar FPS cada 0.5s
                if len(fps_history) > 1:
                    time_span = fps_history[-1] - fps_history[0]
                    if time_span > 0:
                        current_fps = (len(fps_history) - 1) / time_span
                        self.root.after(0, lambda fps=current_fps: self.current_fps.set(f"{fps:.1f} Hz"))
                last_fps_update = loop_start
            
            # Procesar detección - ahora cada frame para mejor confiabilidad
            self._frame_skip_counter += 1
            should_process = self._frame_skip_counter >= self._detection_skip
            
            if should_process:
                self._frame_skip_counter = 0
                self.process_frame_optimized(frame, loop_start)
                # Guardar frame actual para el click handler
                self.current_frame = frame.copy()
            else:
                # Aún así dibujar elementos básicos para continuidad visual
                self.draw_persistent_elements(frame)
            
            # Mostrar frame en tkinter (siempre para fluidez visual)
            self.display_frame_tkinter(frame)
            
            # Control de velocidad de bucle más conservador
            time.sleep(0.005)  # Pequeña pausa para estabilidad
        
    def process_frame_optimized(self, frame, current_time):
        """Procesar frame OPTIMIZADO - reduce operaciones innecesarias"""
        # Detección ArUco (solo cada N frames)
        corners, ids, _ = self.detector.detectMarkers(frame)
        
        if ids is not None:
            ids = ids.flatten()
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Calcular centros de esquinas - solo para las esquinas necesarias
            centers_px = {}
            corner_ids_list = [var.get() for var in self.corner_ids]
            
            for i, mid in enumerate(ids):
                if mid in corner_ids_list:
                    c = corners[i][0]
                    centers_px[int(mid)] = c.mean(axis=0)
            
            # Actualizar homografía con cache mejorado
            H, H_inv = self.update_homography_cache(centers_px, current_time)
            self.CURRENT_H = H
            self.CURRENT_H_INV = H_inv
            
            if H is not None:
                # Dibujar malla solo si cambió la homografía
                self.draw_grid_optimized(frame, H, current_time)
            
            # Procesar robot
            robot_id = self.robot_id.get()
            if (robot_id in ids) and (H_inv is not None):
                self.process_robot_optimized(frame, corners, ids, robot_id, H_inv, current_time)
            else:
                self._robot_detected = False
        else:
            self._robot_detected = False
        
        # Dibujar elementos UI solo cuando sea necesario
        if current_time - self._last_gui_update_time > self._gui_update_interval:
            self.draw_target_marker(frame)
            self.draw_hud_optimized(frame)
            self._last_gui_update_time = current_time
        else:
            # Solo dibujar target si existe (más rápido)
            if self._target_g is not None:
                self.draw_target_marker_simple(frame)
    
    def process_robot_optimized(self, frame, corners, ids, robot_id, H_inv, current_time):
        """Procesar la detección del robot - OPTIMIZADO"""
        idx = list(ids).index(robot_id)
        c_robot = corners[idx][0]
        center_px = c_robot.mean(axis=0)
        px, py = int(center_px[0]), int(center_px[1])
        
        # Usar cache para conversiones si es posible
        cache_key = f"{px}_{py}_{id(H_inv)}"
        if cache_key in self._conversion_cache and (current_time - self._cache_timestamp < 0.1):
            u, v = self._conversion_cache[cache_key]
        else:
            # Coordenadas en la malla
            u, v_top = self.pix_to_grid_uv_fast((px, py), H_inv)
            u = np.clip(u, 0.0, 1.0)
            v = 1.0 - np.clip(v_top, 0.0, 1.0)
            
            # Guardar en cache
            self._conversion_cache[cache_key] = (u, v)
            self._cache_timestamp = current_time
            # Limpiar cache viejo
            if len(self._conversion_cache) > 100:
                self._conversion_cache.clear()
        
        gx = u * self.grid_n.get()
        gy = v * self.grid_n.get()
        
        # Calcular ángulo del robot solo si es necesario
        if self._target_g is not None or (current_time - self._last_gui_update_time > self._gui_update_interval):
            p_dir_img = (c_robot[0] + 0.5*(c_robot[1] - c_robot[0]))
            u0, v0_top = self.pix_to_grid_uv_fast(center_px, H_inv)
            u1, v1_top = self.pix_to_grid_uv_fast(p_dir_img, H_inv)
            
            du, dv = (u1 - u0), (1.0 - v1_top) - (1.0 - v0_top)
            angle_grid = np.degrees(np.arctan2(dv, du))
            self._current_robot_heading = angle_grid
        
        # Visualizar robot
        cv2.circle(frame, (px, py), 8, (0,0,255), -1)
        if current_time - self._last_gui_update_time > self._gui_update_interval:
            cv2.putText(frame, f"R({gx:.1f},{gy:.1f}) {self._current_robot_heading:.0f}°",
                       (px+15, py-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        
        self._last_robot_px = (px, py)
        self._robot_detected = True
        
        # Actualizar estado en GUI solo ocasionalmente
        if current_time - self._last_gui_update_time > self._gui_update_interval:
            pos_text = f"({gx:.2f}, {gy:.2f}) @ {self._current_robot_heading:.1f}°"
            self.root.after(0, lambda: self.robot_position.set(pos_text))
        
        # Control del robot con throttling más conservador
        if self._target_g is not None and (current_time - self._last_command_time > self._command_interval):
            self.goto_controller_step_optimized(gx, gy, self._current_robot_heading, current_time)
            self._last_command_time = current_time
        
    def draw_grid_optimized(self, img, H, current_time):
        """Dibujar malla de homografía - OPTIMIZADO"""
        if H is None:
            return
        
        # Solo redibujar si la homografía cambió significativamente
        if hasattr(self, '_last_H') and self._last_H is not None:
            diff = np.linalg.norm(H - self._last_H)
            if diff < 0.1:  # Umbral de cambio
                return
        
        self._last_H = H.copy()
            
        try:
            N = self.grid_n.get()
            step = 1.0 / N
            
            # Líneas verticales (cada 3 para mejor rendimiento)
            for i in range(0, N+1, 3):
                u = i * step
                src = np.array([[[u, 0.0]], [[u, 1.0]]], dtype=np.float32)
                dst = cv2.perspectiveTransform(src, H)
                p1, p2 = tuple(dst[0,0].astype(int)), tuple(dst[1,0].astype(int))
                cv2.line(img, p1, p2, (0,255,0), 1)
            
            # Líneas horizontales (cada 3)
            for i in range(0, N+1, 3):
                v = i * step
                src = np.array([[[0.0, v]], [[1.0, v]]], dtype=np.float32)
                dst = cv2.perspectiveTransform(src, H)
                p1, p2 = tuple(dst[0,0].astype(int)), tuple(dst[1,0].astype(int))
                cv2.line(img, p1, p2, (0,255,0), 1)

            # Borde
            box = np.array([[[0,0]], [[1,0]], [[1,1]], [[0,1]], [[0,0]]], dtype=np.float32)
            box_t = cv2.perspectiveTransform(box, H).astype(int)
            cv2.polylines(img, [box_t.reshape(-1,2)], isClosed=True, color=(0,180,255), thickness=2)
            
        except Exception as e:
            print(f"[WARN] Error dibujando malla: {e}")
    
    def draw_target_marker_simple(self, img):
        """Dibujar marcador de objetivo simple - MÁS RÁPIDO"""
        if self._target_g is None or self.CURRENT_H is None:
            return

        px = self.grid_to_pix_fast(self._target_g[0], self._target_g[1])
        if px is None:
            return

        x, y = px
        cv2.circle(img, (x, y), 8, (0, 255, 255), 2)
    
    def draw_persistent_elements(self, frame):
        """Dibujar elementos persistentes cuando no se procesa detección completa"""
        # Dibujar malla usando la homografía cacheada
        if self.CURRENT_H is not None:
            self.draw_grid_consistent(frame, self.CURRENT_H)
        
        # Dibujar objetivo si existe
        if self._target_g is not None:
            self.draw_target_marker_simple(frame)
        
        # Dibujar último robot conocido
        if self._last_robot_px is not None:
            cv2.circle(frame, self._last_robot_px, 8, (0,0,255), -1)
    
    def draw_grid_consistent(self, img, H):
        """Dibujar malla de forma consistente - SIN PARPADEO"""
        if H is None:
            return
            
        try:
            N = self.grid_n.get()
            step = 1.0 / N
            
            # Líneas verticales principales (cada 2 para rendimiento)
            for i in range(0, N+1, 2):
                u = i * step
                src = np.array([[[u, 0.0]], [[u, 1.0]]], dtype=np.float32)
                dst = cv2.perspectiveTransform(src, H)
                p1, p2 = tuple(dst[0,0].astype(int)), tuple(dst[1,0].astype(int))
                cv2.line(img, p1, p2, (0,255,0), 1)
            
            # Líneas horizontales principales (cada 2)
            for i in range(0, N+1, 2):
                v = i * step
                src = np.array([[[0.0, v]], [[1.0, v]]], dtype=np.float32)
                dst = cv2.perspectiveTransform(src, H)
                p1, p2 = tuple(dst[0,0].astype(int)), tuple(dst[1,0].astype(int))
                cv2.line(img, p1, p2, (0,255,0), 1)

            # Borde siempre visible y destacado
            box = np.array([[[0,0]], [[1,0]], [[1,1]], [[0,1]], [[0,0]]], dtype=np.float32)
            box_t = cv2.perspectiveTransform(box, H).astype(int)
            cv2.polylines(img, [box_t.reshape(-1,2)], isClosed=True, color=(0,180,255), thickness=2)
            
            # Ejes de referencia
            ax = np.array([[[0,0]], [[0.15,0]]], dtype=np.float32)
            ay = np.array([[[0,0]], [[0,0.15]]], dtype=np.float32)
            ax_t = cv2.perspectiveTransform(ax, H).astype(int)
            ay_t = cv2.perspectiveTransform(ay, H).astype(int)
            cv2.arrowedLine(img, tuple(ax_t[0,0]), tuple(ax_t[1,0]), (255,80,80), 2, tipLength=0.1)
            cv2.arrowedLine(img, tuple(ay_t[0,0]), tuple(ay_t[1,0]), (80,80,255), 2, tipLength=0.1)
            
            # Etiquetas de ejes (solo ocasionalmente para rendimiento)
            if not hasattr(self, '_axis_label_counter'):
                self._axis_label_counter = 0
            self._axis_label_counter += 1
            if self._axis_label_counter % 20 == 0:  # Solo cada 20 frames
                cv2.putText(img, "X", tuple(ax_t[1,0]+[5,-5]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,80,80), 1)
                cv2.putText(img, "Y", tuple(ay_t[1,0]+[5,-5]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80,80,255), 1)
            
        except Exception as e:
            print(f"[WARN] Error dibujando malla: {e}")
    
    def draw_grid_optimized(self, img, H, current_time):
        """Mantener compatibilidad - redirige a método consistente"""
        self.draw_grid_consistent(img, H)
    
    def process_frame_optimized(self, frame, current_time):
        """Procesar frame BALANCEADO - mejor detección sin parpadeo"""
        # Detección ArUco siempre activa
        corners, ids, _ = self.detector.detectMarkers(frame)
        
        if ids is not None:
            ids = ids.flatten()
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Calcular centros de esquinas - solo para las esquinas necesarias
            centers_px = {}
            corner_ids_list = [var.get() for var in self.corner_ids]
            
            for i, mid in enumerate(ids):
                if mid in corner_ids_list:
                    c = corners[i][0]
                    centers_px[int(mid)] = c.mean(axis=0)
            
            # Actualizar homografía con cache más conservador
            H, H_inv = self.update_homography_cache_balanced(centers_px, current_time)
            self.CURRENT_H = H
            self.CURRENT_H_INV = H_inv
            
            # Siempre dibujar la malla completa cuando tenemos homografía válida
            if H is not None:
                self.draw_grid_consistent(frame, H)
            
            # Procesar robot
            robot_id = self.robot_id.get()
            if (robot_id in ids) and (H_inv is not None):
                self.process_robot_optimized(frame, corners, ids, robot_id, H_inv, current_time)
            else:
                self._robot_detected = False
        else:
            self._robot_detected = False
            # Mantener malla usando homografía cacheada
            if self.CURRENT_H is not None:
                self.draw_grid_consistent(frame, self.CURRENT_H)
        
        # Dibujar elementos UI controlados por frecuencia
        if current_time - self._last_gui_update_time > self._gui_update_interval:
            self.draw_target_marker(frame)
            self.draw_hud_optimized(frame)
            self._last_gui_update_time = current_time
        else:
            # Dibujar elementos mínimos para continuidad
            if self._target_g is not None:
                self.draw_target_marker_simple(frame)
    
    def update_homography_cache_balanced(self, centers_px, current_time):
        """Actualizar cache de homografía - BALANCEADO para mejor detección"""
        # Recalcular con más frecuencia para mejor seguimiento
        if current_time - self._homography_cache["timestamp"] < 0.1:  # Cada 100ms
            if self._homography_cache["H"] is not None:
                return self._homography_cache["H"], self._homography_cache["H_inv"]
        
        if len(centers_px) == 4:
            try:
                pts_dst = self.order_quad([centers_px[k] for k in centers_px.keys()])
                pts_src = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]], dtype=np.float32)
                H = cv2.getPerspectiveTransform(pts_src, pts_dst)
                H_inv = np.linalg.inv(H)
                
                self._homography_cache.update({
                    "H": H,
                    "H_inv": H_inv,
                    "timestamp": current_time
                })
                
                return H, H_inv
            except Exception as e:
                print(f"[WARN] Error calculando homografía: {e}")
        
        return self._homography_cache["H"], self._homography_cache["H_inv"]
    
    def draw_grid_reliable(self, img, H):
        """Dibujar malla completa - para redibujado periódico"""
        if H is None:
            return
            
        try:
            N = self.grid_n.get()
            step = 1.0 / N
            
            # Líneas verticales (cada 2 para balance rendimiento/visibilidad)
            for i in range(0, N+1, 2):
                u = i * step
                src = np.array([[[u, 0.0]], [[u, 1.0]]], dtype=np.float32)
                dst = cv2.perspectiveTransform(src, H)
                p1, p2 = tuple(dst[0,0].astype(int)), tuple(dst[1,0].astype(int))
                cv2.line(img, p1, p2, (0,255,0), 1)
            
            # Líneas horizontales (cada 2)
            for i in range(0, N+1, 2):
                v = i * step
                src = np.array([[[0.0, v]], [[1.0, v]]], dtype=np.float32)
                dst = cv2.perspectiveTransform(src, H)
                p1, p2 = tuple(dst[0,0].astype(int)), tuple(dst[1,0].astype(int))
                cv2.line(img, p1, p2, (0,255,0), 1)

            # Borde siempre visible
            box = np.array([[[0,0]], [[1,0]], [[1,1]], [[0,1]], [[0,0]]], dtype=np.float32)
            box_t = cv2.perspectiveTransform(box, H).astype(int)
            cv2.polylines(img, [box_t.reshape(-1,2)], isClosed=True, color=(0,180,255), thickness=2)
            
            # Ejes siempre visibles
            ax = np.array([[[0,0]], [[0.15,0]]], dtype=np.float32)
            ay = np.array([[[0,0]], [[0,0.15]]], dtype=np.float32)
            ax_t = cv2.perspectiveTransform(ax, H).astype(int)
            ay_t = cv2.perspectiveTransform(ay, H).astype(int)
            cv2.arrowedLine(img, tuple(ax_t[0,0]), tuple(ax_t[1,0]), (255,80,80), 2, tipLength=0.1)
            cv2.arrowedLine(img, tuple(ay_t[0,0]), tuple(ay_t[1,0]), (80,80,255), 2, tipLength=0.1)
            
        except Exception as e:
            print(f"[WARN] Error dibujando malla: {e}")
    
    def draw_grid_simple(self, img, H):
        """Dibujar solo el borde de la malla - para frames intermedios"""
        if H is None:
            return
            
        try:
            # Solo borde para mantener referencia visual
            box = np.array([[[0,0]], [[1,0]], [[1,1]], [[0,1]], [[0,0]]], dtype=np.float32)
            box_t = cv2.perspectiveTransform(box, H).astype(int)
            cv2.polylines(img, [box_t.reshape(-1,2)], isClosed=True, color=(0,180,255), thickness=2)
        except Exception as e:
            print(f"[draw_grid_simple] Error: {e}")
            
    def draw_grid_optimized(self, img, H, current_time):
        """Dibujar malla de homografía - BALANCEADO"""
        if H is None:
            return
        
        # Control de frecuencia de redibujado completo
        if not hasattr(self, '_last_full_grid_time'):
            self._last_full_grid_time = 0
            
        # Redibujado completo cada 200ms
        if current_time - self._last_full_grid_time > 0.2:
            self.draw_grid_reliable(img, H)
            self._last_full_grid_time = current_time
        else:
            # Solo borde para frames intermedios
            self.draw_grid_simple(img, H)
    
    def draw_hud_optimized(self, img):
        """Dibujar HUD de información - OPTIMIZADO"""
        # Solo información esencial
        hud_lines = [
            f"Robot ID: {self.robot_id.get()}",
            "Click para objetivo"
        ]
        
        y_start = 25
        for i, line in enumerate(hud_lines):
            y = y_start + i * 20
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    def update_homography_cache(self, centers_px, current_time):
        """Actualizar cache de homografía - MEJORADO"""
        # Recalcular menos frecuentemente
        if current_time - self._homography_cache["timestamp"] < 0.2:  # Cada 200ms
            if self._homography_cache["H"] is not None:
                return self._homography_cache["H"], self._homography_cache["H_inv"]
        
        if len(centers_px) == 4:
            try:
                pts_dst = self.order_quad([centers_px[k] for k in centers_px.keys()])
                pts_src = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]], dtype=np.float32)
                H = cv2.getPerspectiveTransform(pts_src, pts_dst)
                H_inv = np.linalg.inv(H)
                
                self._homography_cache.update({
                    "H": H,
                    "H_inv": H_inv,
                    "timestamp": current_time
                })
                
                return H, H_inv
            except:
                pass
        
        return self._homography_cache["H"], self._homography_cache["H_inv"]
    
    def order_quad(self, pts):
        """Ordenar cuadrilátero"""
        pts = np.asarray(pts, dtype=np.float32)
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        return pts[sorted_indices]
    
    def draw_target_marker(self, img):
        """Dibujar marcador de objetivo"""
        if self._target_g is None or self.CURRENT_H is None:
            return

        px = self.grid_to_pix_fast(self._target_g[0], self._target_g[1])
        if px is None:
            return

        x, y = px
        cv2.circle(img, (x, y), 12, (0, 255, 255), 3)
        cv2.line(img, (x-16, y), (x+16, y), (0, 255, 255), 2)
        cv2.line(img, (x, y-16), (x, y+16), (0, 255, 255), 2)
        cv2.putText(img, f"Objetivo ({self._target_g[0]:.1f},{self._target_g[1]:.1f})",
                    (x+15, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        
        # Línea robot -> objetivo
        if self._last_robot_px is not None:
            cv2.line(img, self._last_robot_px, px, (0, 200, 255), 2)
    
    def display_frame_tkinter(self, frame):
        """Mostrar frame en el widget de tkinter - OPTIMIZADO"""
        try:
            # Redimensionar frame para que quepa en el widget
            widget_width = self.video_label.winfo_width()
            widget_height = self.video_label.winfo_height()
            
            # Si el widget aún no tiene tamaño, usar valores por defecto
            if widget_width <= 1:
                widget_width = 640
            if widget_height <= 1:
                widget_height = 480
                
            frame_height, frame_width = frame.shape[:2]
            
            # Calcular escala manteniendo aspecto
            scale_x = widget_width / frame_width
            scale_y = widget_height / frame_height
            scale = min(scale_x, scale_y)
            
            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)
            
            # Redimensionar frame solo si es necesario
            if new_width > 0 and new_height > 0:
                # Usar interpolación más rápida
                resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
                
                # Convertir de BGR a RGB
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                
                # Convertir a PIL Image
                pil_image = Image.fromarray(rgb_frame)
                
                # Convertir a PhotoImage para tkinter
                photo = ImageTk.PhotoImage(pil_image)
                
                # Actualizar el label en el hilo principal
                self.root.after(0, self.update_video_label, photo)
                
        except Exception as e:
            print(f"[ERROR] Error mostrando frame: {e}")
    
    def update_video_label(self, photo):
        """Actualizar el label del video en el hilo principal"""
        try:
            self.video_label.configure(image=photo, text="")
            self.video_label.image = photo  # Mantener referencia
        except Exception as e:
            print(f"[ERROR] Error actualizando video label: {e}")
    
    def pix_to_grid_uv_fast(self, pt_pix, H_inv):
        """Conversión rápida pixel -> UV"""
        if H_inv is None:
            return 0.0, 0.0
        try:
            x, y = pt_pix[0], pt_pix[1]
            w = H_inv[2,0]*x + H_inv[2,1]*y + H_inv[2,2]
            if abs(w) < 1e-8:
                return 0.0, 0.0
            u = (H_inv[0,0]*x + H_inv[0,1]*y + H_inv[0,2]) / w
            v = (H_inv[1,0]*x + H_inv[1,1]*y + H_inv[1,2]) / w
            return float(u), float(v)
        except:
            return 0.0, 0.0
    
    def grid_to_pix_fast(self, gx, gy):
        """Conversión rápida grid -> pixel"""
        if self.CURRENT_H is None:
            return None
        try:
            u = float(gx) / float(self.grid_n.get())
            v_top = 1.0 - (float(gy) / float(self.grid_n.get()))
            
            H = self.CURRENT_H
            w = H[2,0]*u + H[2,1]*v_top + H[2,2]
            if abs(w) < 1e-8:
                return None
            x = (H[0,0]*u + H[0,1]*v_top + H[0,2]) / w
            y = (H[1,0]*u + H[1,1]*v_top + H[1,2]) / w
            return int(x), int(y)
        except:
            return None
    
    # ==========================
    #  Métodos de control del robot OPTIMIZADOS
    # ==========================
    def set_target(self, gx_star, gy_star, current_heading_deg=None):
        """Establecer objetivo"""
        self._target_g = (float(gx_star), float(gy_star))
        
        if not self.orient_mode.get() and current_heading_deg is not None:
            self._hold_heading_rad = math.radians(current_heading_deg)
        
        target_text = f"({self._target_g[0]:.2f}, {self._target_g[1]:.2f})"
        self.root.after(0, lambda: self.target_position.set(target_text))
        
        print(f"[GOTO] Nuevo objetivo: {target_text}")
    
    def clear_target(self):
        """Limpiar objetivo"""
        self._target_g = None
        self._hold_heading_rad = None
        self.stop_robot()
        self.root.after(0, lambda: self.target_position.set("Sin objetivo"))
    
    def manual_command(self, vx, vy, w):
        """Enviar comando manual al robot"""
        if not self.firebase_connected:
            messagebox.showwarning("Sin Conexión", "Conecta a Firebase primero")
            return
        
        try:
            # Limpiar objetivo automático si existe
            self._target_g = None
            self._hold_heading_rad = None
            
            # Enviar comando directamente
            self.firebase_queue.put_nowait({
                'type': 'movement',
                'vx': int(vx),
                'vy': int(vy)
            })
            self.firebase_queue.put_nowait({
                'type': 'rotation', 
                'w': int(w)
            })
            
            print(f"[MANUAL] vx={vx} vy={vy} w={w}")
            
        except queue.Full:
            print("[MANUAL][WARN] Cola Firebase llena")
        except Exception as e:
            print(f"[MANUAL][ERROR] {e}")
    
    def stop_robot(self):
        """Parar robot - NO BLOQUEA"""
        if not self.firebase_connected:
            return
        try:
            # Enviar comando a la cola en lugar de directamente
            self.firebase_queue.put_nowait({'type': 'stop'})
            print("[GOTO] Robot detenido")
        except queue.Full:
            print("[GOTO][WARN] Cola Firebase llena, comando descartado")
    
    def _send_cmd_optimized(self, vx, vy, w):
        """Enviar comando al robot - NO BLOQUEA"""
        if not self.firebase_connected:
            return
        try:
            # Enviar movimiento
            self.firebase_queue.put_nowait({
                'type': 'movement',
                'vx': int(vx),
                'vy': int(vy)
            })
            # Enviar rotación
            self.firebase_queue.put_nowait({
                'type': 'rotation',
                'w': int(w)
            })
        except queue.Full:
            print("[GOTO][WARN] Cola Firebase llena, comando descartado")
    
    def _sat(self, x, lim):
        """Saturar valor"""
        return max(-lim, min(lim, x))
    
    def goto_controller_step_optimized(self, gx, gy, heading_deg, current_time):
        """Paso del controlador goto - OPTIMIZADO"""
        if self._target_g is None:
            return

        ex = self._target_g[0] - gx
        ey = self._target_g[1] - gy
        dist = math.hypot(ex, ey)

        if dist <= self.pos_tolerance.get():
            self._send_cmd_optimized(0, 0, 0)
            if hasattr(self, '_target_reached_logged') and not self._target_reached_logged:
                print(f"[GOTO] Objetivo alcanzado: dist={dist:.3f}")
                self._target_reached_logged = True
            return
        
        self._target_reached_logged = False

        # Control proporcional lineal
        vx_cmd = self.kp_lin.get() * ex
        vy_cmd = self.kp_lin.get() * ey

        # Limitar norma del vector
        norm = math.hypot(vx_cmd, vy_cmd)
        cmd_max = self.cmd_max.get()
        if norm > cmd_max:
            scale = cmd_max / (norm + 1e-9)
            vx_cmd *= scale
            vy_cmd *= scale

        th = math.radians(heading_deg)

        # Comando de giro
        if self.orient_mode.get():
            if norm > 1e-3:
                th_des = math.atan2(vy_cmd, vx_cmd)
                e_th = math.atan2(math.sin(th_des - th), math.cos(th_des - th))
                w_cmd = self.kp_w_face.get() * e_th
            else:
                w_cmd = 0.0
        else:
            if self._hold_heading_rad is None:
                self._hold_heading_rad = th
            e_th = math.atan2(math.sin(self._hold_heading_rad - th), math.cos(self._hold_heading_rad - th))
            w_cmd = self.kp_w_hold.get() * e_th

        w_cmd = self._sat(w_cmd, self.w_max.get())

        # Enviar comando de forma no bloqueante
        self._send_cmd_optimized(round(vx_cmd), round(vy_cmd), round(w_cmd))
    
    # ==========================
    #  Métodos de cierre
    # ==========================
    def on_closing(self):
        """Manejar cierre de aplicación"""
        if self.running:
            self.stop_system()
        
        # Detener Firebase
        self.disconnect_firebase()
        
        # Guardar configuración automáticamente
        try:
            self.save_config()
        except:
            pass
            
        self.root.destroy()

# ==========================
#  Función principal
# ==========================
def main():
    """Función principal de la aplicación"""
    # Configurar estilo moderno
    root = tk.Tk()
    
    # Configurar estilo ttk
    style = ttk.Style()
    if "vista" in style.theme_names():
        style.theme_use("vista")
    elif "clam" in style.theme_names():
        style.theme_use("clam")
    
    # Colores personalizados
    style.configure('Accent.TButton', 
                   font=('Arial', 10, 'bold'),
                   focuscolor='red')
    
    # Crear aplicación
    app = ArucoRobotGUI(root)
    
    # Centrar ventana
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Iniciar aplicación
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n[INFO] Aplicación cerrada por usuario")
    except Exception as e:
        print(f"[ERROR] Error inesperado: {e}")
        messagebox.showerror("Error Fatal", f"Error inesperado en la aplicación:\n{e}")

if __name__ == "__main__":
    main()