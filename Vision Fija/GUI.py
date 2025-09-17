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
        self.config_file = "robot_config.json"
        
        # Variables para la visualización en tkinter
        self.video_label = None
        self.current_frame = None
        self.display_frame = None
        
        # Variables de optimización de procesamiento
        self.detection_scale = 0.5  # Procesar detección en resolución reducida
        self.display_scale_factor = 1.0
        self.process_every_n = 3  # Procesar detección cada N frames
        self.display_every_n = 2  # Mostrar cada N frames
        self.frame_counter = 0
        self.detection_counter = 0
        
        # Variables de configuración
        self.camera_index = tk.IntVar(value=0)  # Cambio a 0 por defecto
        self.camera_width = tk.IntVar(value=1280)  # Reducido de 1920
        self.camera_height = tk.IntVar(value=720)   # Reducido de 1080
        self.camera_fps = tk.IntVar(value=30)
        self.robot_id = tk.IntVar(value=4)
        self.corner_ids = [tk.IntVar(value=1), tk.IntVar(value=2), tk.IntVar(value=5), tk.IntVar(value=3)]
        self.grid_n = tk.IntVar(value=10)
        
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
        
        # Variables del procesador de video
        self._target_g = None
        self._hold_heading_rad = None
        self.CURRENT_H = None
        self.CURRENT_H_INV = None
        self._last_robot_px = None
        self._current_robot_heading = 0.0
        
        # Variables de optimización ArUco mejoradas
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # Optimizaciones ArUco críticas - Compatibles con todas las versiones
        try:
            self.aruco_params.minMarkerPerimeterRate = 0.03
            self.aruco_params.maxMarkerPerimeterRate = 4.0
            self.aruco_params.polygonalApproxAccuracyRate = 0.05
            self.aruco_params.minCornerDistanceRate = 0.05
            self.aruco_params.minDistanceToBorder = 3
            self.aruco_params.minMarkerDistanceRate = 0.05
            self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
            self.aruco_params.cornerRefinementWinSize = 3
            self.aruco_params.cornerRefinementMaxIterations = 10
            self.aruco_params.markerBorderBits = 1
            self.aruco_params.maxErroneousBitsInBorderRate = 0.35
            self.aruco_params.minOtsuStdDev = 5.0
            self.aruco_params.errorCorrectionRate = 0.6
            
            # Parámetros que pueden no existir en todas las versiones
            if hasattr(self.aruco_params, 'perspectiveRemovePixelPerKsize'):
                self.aruco_params.perspectiveRemovePixelPerKsize = 4
            if hasattr(self.aruco_params, 'perspectiveRemoveIgnoredMarginPerCell'):
                self.aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.1
            if hasattr(self.aruco_params, 'adaptiveThreshWinSizeMin'):
                self.aruco_params.adaptiveThreshWinSizeMin = 3
            if hasattr(self.aruco_params, 'adaptiveThreshWinSizeMax'):
                self.aruco_params.adaptiveThreshWinSizeMax = 23
            if hasattr(self.aruco_params, 'adaptiveThreshWinSizeStep'):
                self.aruco_params.adaptiveThreshWinSizeStep = 10
                
        except AttributeError as e:
            print(f"[WARN] Algunos parámetros ArUco no disponibles: {e}")
            # Configuración mínima compatible
            self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Cache optimizado
        self._homography_cache = {
            "H": None, 
            "H_inv": None, 
            "timestamp": 0, 
            "corner_positions": None,
            "valid_duration": 0.5  # Cache válido por más tiempo
        }
        
        # Cache de frames para evitar conversiones innecesarias
        self._frame_cache = {
            "original": None,
            "detection": None,
            "display": None,
            "timestamp": 0
        }
        
        # Variables de estado del robot (cache)
        self._robot_state = {
            "detected": False,
            "position": (0, 0),
            "heading": 0,
            "timestamp": 0,
            "px_position": None
        }
        
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
        style.theme_use('clam')
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
        
        # Pestaña 4: Optimización
        optim_frame = ttk.Frame(notebook)
        notebook.add(optim_frame, text="Optimización")
        self.create_optimization_config(optim_frame)
        
        # Pestaña 5: Firebase
        firebase_frame = ttk.Frame(notebook)
        notebook.add(firebase_frame, text="Firebase")
        self.create_firebase_config(firebase_frame)
        
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
        
        # Resoluciones predefinidas optimizadas
        res_group = ttk.LabelFrame(parent, text="Resoluciones Optimizadas", padding=10)
        res_group.pack(fill=tk.X, padx=10, pady=5)
        
        res_buttons = [
            ("720p (Recomendado)", 1280, 720),
            ("480p (Rápido)", 640, 480),
            ("1080p (Lento)", 1920, 1080)
        ]
        
        for i, (name, w, h) in enumerate(res_buttons):
            btn = ttk.Button(res_group, text=name, 
                           command=lambda w=w, h=h: self.set_resolution(w, h))
            btn.grid(row=0, column=i, padx=2, pady=5)
            
    def create_optimization_config(self, parent):
        """Configuración de optimización de rendimiento"""
        # Variables de optimización
        self.detection_scale_var = tk.DoubleVar(value=0.5)
        self.process_every_var = tk.IntVar(value=3)
        self.display_every_var = tk.IntVar(value=2)
        self.corner_refinement_var = tk.BooleanVar(value=False)
        
        # Escala de detección
        scale_group = ttk.LabelFrame(parent, text="Escala de Detección", padding=10)
        scale_group.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(scale_group, text="Escala (menor = más rápido):").grid(row=0, column=0, sticky=tk.W, pady=2)
        scale_scale = ttk.Scale(scale_group, from_=0.2, to=1.0, variable=self.detection_scale_var, 
                               orient=tk.HORIZONTAL, length=150)
        scale_scale.grid(row=0, column=1, padx=(10,5), pady=2)
        self.scale_label = ttk.Label(scale_group, text=f"{self.detection_scale_var.get():.2f}")
        self.scale_label.grid(row=0, column=2, pady=2)
        
        self.detection_scale_var.trace_add('write', self.update_detection_scale)
        
        # Frecuencia de procesamiento
        freq_group = ttk.LabelFrame(parent, text="Frecuencia de Procesamiento", padding=10)
        freq_group.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(freq_group, text="Procesar cada N frames:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(freq_group, from_=1, to=10, textvariable=self.process_every_var, width=10).grid(row=0, column=1, padx=(10,0), pady=2)
        
        ttk.Label(freq_group, text="Mostrar cada N frames:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(freq_group, from_=1, to=5, textvariable=self.display_every_var, width=10).grid(row=1, column=1, padx=(10,0), pady=2)
        
        # Opciones ArUco
        aruco_group = ttk.LabelFrame(parent, text="Optimizaciones ArUco", padding=10)
        aruco_group.pack(fill=tk.X, padx=10, pady=5)
        
        corner_check = ttk.Checkbutton(aruco_group, text="Refinamiento de esquinas (más lento)", 
                                     variable=self.corner_refinement_var,
                                     command=self.update_aruco_params)
        corner_check.pack(pady=2)
        
        # Aplicar configuraciones
        apply_btn = ttk.Button(parent, text="Aplicar Optimizaciones", command=self.apply_optimizations)
        apply_btn.pack(pady=10)
        
    def update_detection_scale(self, *args):
        """Actualizar escala de detección"""
        self.scale_label.configure(text=f"{self.detection_scale_var.get():.2f}")
        
    def update_aruco_params(self):
        """Actualizar parámetros ArUco"""
        if self.corner_refinement_var.get():
            self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        else:
            self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
            
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
    def apply_optimizations(self):
        """Aplicar configuraciones de optimización"""
        self.detection_scale = self.detection_scale_var.get()
        self.process_every_n = self.process_every_var.get()
        self.display_every_n = self.display_every_var.get()
        self.update_aruco_params()
        messagebox.showinfo("Optimizaciones", "Configuraciones aplicadas correctamente")
        
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
        video_title = ttk.Label(parent, text="Vista de Camara - Sistema Optimizado", style='Title.TLabel')
        video_title.pack(pady=(10, 5))
        
        # Frame para video
        video_container = ttk.Frame(parent, relief=tk.SUNKEN, borderwidth=2)
        video_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Label para mostrar el video
        self.video_label = ttk.Label(video_container, 
                                   text="Vista de camara optimizada\n\nClick para establecer objetivo\n\nSistema configurado para máximo rendimiento", 
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
            ("FPS Real:", self.current_fps),
            ("Posición del Robot:", self.robot_position),
            ("Objetivo Actual:", self.target_position),
            ("Estado Firebase:", self.connection_status)
        ]
        
        for i, (label_text, var) in enumerate(status_items):
            ttk.Label(status_frame, text=label_text).grid(row=i//2, column=(i%2)*2, sticky=tk.W, padx=5, pady=2)
            status_label = ttk.Label(status_frame, textvariable=var, style='Status.TLabel')
            status_label.grid(row=i//2, column=(i%2)*2+1, sticky=tk.W, padx=5, pady=2)
    
    def on_video_click(self, event):
        """Manejar click en el video optimizado"""
        if self.CURRENT_H_INV is None or not self.running or self.current_frame is None:
            return
        
        try:
            # Obtener coordenadas relativas al widget
            widget_x = event.x
            widget_y = event.y
            
            # Calcular la escala y offset del video en el widget
            widget_width = self.video_label.winfo_width()
            widget_height = self.video_label.winfo_height()
            
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
                self.set_target(gx, gy, self._current_robot_heading)
                print(f"[CLICK] Nuevo objetivo en ({gx:.2f}, {gy:.2f})")
        except Exception as e:
            print(f"[ERROR] Error procesando click: {e}")
        
    # ==========================
    #  Métodos de configuración optimizados
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
        """Guardar configuración actual incluyendo optimizaciones"""
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
            'optimization': {
                'detection_scale': getattr(self, 'detection_scale_var', tk.DoubleVar(value=0.5)).get(),
                'process_every': getattr(self, 'process_every_var', tk.IntVar(value=3)).get(),
                'display_every': getattr(self, 'display_every_var', tk.IntVar(value=2)).get(),
                'corner_refinement': getattr(self, 'corner_refinement_var', tk.BooleanVar(value=False)).get()
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
        """Cargar configuración desde archivo incluyendo optimizaciones"""
        if not os.path.exists(self.config_file):
            return
            
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                
            # Cargar configuración de cámara
            if 'camera' in config:
                cam = config['camera']
                self.camera_index.set(cam.get('index', 0))
                self.camera_width.set(cam.get('width', 1280))
                self.camera_height.set(cam.get('height', 720))
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
                
            # Cargar configuración de optimización
            if 'optimization' in config:
                opt = config['optimization']
                if hasattr(self, 'detection_scale_var'):
                    self.detection_scale_var.set(opt.get('detection_scale', 0.5))
                if hasattr(self, 'process_every_var'):
                    self.process_every_var.set(opt.get('process_every', 3))
                if hasattr(self, 'display_every_var'):
                    self.display_every_var.set(opt.get('display_every', 2))
                if hasattr(self, 'corner_refinement_var'):
                    self.corner_refinement_var.set(opt.get('corner_refinement', False))
                    
                self.detection_scale = opt.get('detection_scale', 0.5)
                self.process_every_n = opt.get('process_every', 3)
                self.display_every_n = opt.get('display_every', 2)
                
            # Cargar configuración de Firebase
            if 'firebase' in config:
                fb = config['firebase']
                self.firebase_url.set(fb.get('url', ''))
                self.firebase_path.set(fb.get('path', ''))
                self.cred_path.set(fb.get('cred_path', ''))
                
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la configuración:\n{e}")
            
    # ==========================
    #  Métodos de Firebase
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
            
            self.connection_status.set("Conectado")
            messagebox.showinfo("Éxito", "Conectado a Firebase correctamente")
            
        except Exception as e:
            self.connection_status.set("Error de conexión")
            messagebox.showerror("Error Firebase", f"No se pudo conectar a Firebase:\n{e}")
            
    def disconnect_firebase(self):
        """Desconectar Firebase"""
        self.instr_ref = None
        self.connection_status.set("Desconectado")
        
    # ==========================
    #  Métodos del sistema principal optimizados
    # ==========================
    def toggle_system(self):
        """Iniciar/parar el sistema"""
        if not self.running:
            self.start_system()
        else:
            self.stop_system()
            
    def start_system(self):
        """Iniciar el sistema de detección y control optimizado"""
        try:
            # Abrir cámara
            self.cap = self.open_camera()
            
            # Reiniciar contadores
            self.frame_counter = 0
            self.detection_counter = 0
            
            # Cambiar botón
            self.start_btn.configure(text="PARAR SISTEMA")
            
            # Iniciar hilo de cámara
            self.running = True
            self.camera_thread = threading.Thread(target=self.camera_loop_optimized, daemon=True)
            self.camera_thread.start()
            
            messagebox.showinfo("Sistema Iniciado", "Sistema optimizado iniciado correctamente")
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo iniciar el sistema:\n{e}")
            
    def stop_system(self):
        """Parar el sistema"""
        self.running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
            
        # Limpiar caches
        self._frame_cache = {
            "original": None,
            "detection": None,
            "display": None,
            "timestamp": 0
        }
        
        # Limpiar el video label
        self.video_label.configure(image='', text="Vista de camara optimizada\n\nClick para establecer objetivo\n\nSistema configurado para máximo rendimiento")
        self.current_frame = None
        
        self.start_btn.configure(text="INICIAR SISTEMA")
        self.current_fps.set("0.0 Hz")
        self.robot_position.set("No detectado")
        
        messagebox.showinfo("Sistema Detenido", "Sistema detenido correctamente")
        
    def open_camera(self):
        """Abrir cámara con configuración optimizada"""
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2]
        
        cap = None
        for backend in backends:
            cap = cv2.VideoCapture(self.camera_index.get(), backend)
            if cap.isOpened():
                break
            if cap:
                cap.release()
                
        if not cap or not cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara")
            
        # Configurar cámara para máximo rendimiento
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo para reducir latencia
        cap.set(cv2.CAP_PROP_FPS, self.camera_fps.get())
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width.get())
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height.get())
        
        # Configuraciones adicionales para rendimiento
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Reducir auto-exposición
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Desactivar auto-foco
        
        # Intentar desactivar procesamiento adicional
        try:
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # Auto white balance
            cap.set(cv2.CAP_PROP_SETTINGS, 0)  # Ventana de configuración
        except:
            pass  # No todas las cámaras soportan estas propiedades
        
        return cap
        
    def camera_loop_optimized(self):
        """Loop principal de la cámara altamente optimizado"""
        fps_history = deque(maxlen=30)
        fps_counter = 0
        current_fps = 0.0
        misses = 0
        MAX_MISSES = 10
        
        # Variables de timing
        last_detection_time = 0
        last_display_time = 0
        
        # Pre-allocar arrays para mejor rendimiento
        detection_frame = None
        
        while self.running:
            loop_start = time.time()
            
            # Capturar frame
            ret, frame = self.cap.read()
            if not ret or frame is None:
                misses += 1
                if misses > MAX_MISSES:
                    print("[WARN] Reintentando abrir cámara...")
                    if self.cap:
                        self.cap.release()
                    time.sleep(0.1)
                    try:
                        self.cap = self.open_camera()
                        misses = 0
                    except:
                        self.root.after(0, lambda: messagebox.showerror("Error", "Se perdió la conexión con la cámara"))
                        break
                continue
            misses = 0
            
            self.frame_counter += 1
            current_time = time.time()
            
            # Calcular FPS de forma más eficiente
            fps_counter += 1
            if fps_counter % 15 == 0:  # Actualizar cada 15 frames
                fps_history.append(current_time)
                if len(fps_history) > 1:
                    time_span = fps_history[-1] - fps_history[0]
                    if time_span > 0:
                        current_fps = (len(fps_history) - 1) / time_span
                        self.root.after(0, lambda: self.current_fps.set(f"{current_fps:.1f} Hz"))
            
            # Guardar frame actual para clicks
            self.current_frame = frame
            
            # Procesamiento de detección (cada N frames)
            should_detect = (self.detection_counter % self.process_every_n == 0)
            if should_detect:
                self.detection_counter = 0
                self.process_frame_optimized(frame, current_time)
            self.detection_counter += 1
            
            # Mostrar frame (cada N frames)
            should_display = (self.frame_counter % self.display_every_n == 0)
            if should_display and (current_time - last_display_time) > 0.033:  # Máximo 30 FPS display
                self.display_frame_tkinter_optimized(frame)
                last_display_time = current_time
            
            # Control de timing para evitar sobrecargar el CPU
            elapsed = time.time() - loop_start
            if elapsed < 0.001:  # Mínimo 1ms por loop
                time.sleep(0.001 - elapsed)
        
    def process_frame_optimized(self, frame, current_time):
        """Procesar frame para detección ArUco altamente optimizado"""
        try:
            # Crear frame de detección en resolución reducida
            if self.detection_scale < 1.0:
                h, w = frame.shape[:2]
                new_h, new_w = int(h * self.detection_scale), int(w * self.detection_scale)
                detection_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                scale_factor = 1.0 / self.detection_scale
            else:
                detection_frame = frame
                scale_factor = 1.0
            
            # Detección ArUco optimizada
            corners, ids, _ = self.detector.detectMarkers(detection_frame)
            
            if ids is not None and len(ids) > 0:
                ids = ids.flatten()
                
                # Escalar coordenadas si es necesario
                if scale_factor != 1.0:
                    corners = [corner * scale_factor for corner in corners]
                
                # Dibujar marcadores solo si es necesario
                if self.frame_counter % (self.process_every_n * 2) == 0:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                
                # Procesar esquinas para homografía
                centers_px = {}
                corner_ids_list = [var.get() for var in self.corner_ids]
                
                for i, mid in enumerate(ids):
                    if mid in corner_ids_list:
                        c = corners[i][0]
                        centers_px[int(mid)] = c.mean(axis=0)
                
                # Actualizar homografía (cache optimizado)
                H, H_inv = self.update_homography_cache_optimized(centers_px, current_time)
                self.CURRENT_H = H
                self.CURRENT_H_INV = H_inv
                
                # Dibujar malla solo ocasionalmente para performance
                if H is not None and self.frame_counter % (self.process_every_n * 3) == 0:
                    self.draw_grid_homography_optimized(frame, H)
                
                # Procesar robot
                robot_id = self.robot_id.get()
                if (robot_id in ids) and (H_inv is not None):
                    self.process_robot_optimized(frame, corners, ids, robot_id, H_inv, current_time)
            
            # Dibujar elementos adicionales (menos frecuentemente)
            if self.frame_counter % self.process_every_n == 0:
                self.draw_target_marker_optimized(frame)
                if self.frame_counter % (self.process_every_n * 2) == 0:
                    self.draw_hud_optimized(frame)
                    
        except Exception as e:
            print(f"[ERROR] Error en procesamiento optimizado: {e}")
        
    def process_robot_optimized(self, frame, corners, ids, robot_id, H_inv, current_time):
        """Procesar la detección del robot de forma optimizada"""
        try:
            idx = list(ids).index(robot_id)
            c_robot = corners[idx][0]
            center_px = c_robot.mean(axis=0)
            px, py = int(center_px[0]), int(center_px[1])
            
            # Coordenadas en la malla (cálculo optimizado)
            u, v_top = self.pix_to_grid_uv_fast(center_px, H_inv)
            u = np.clip(u, 0.0, 1.0)
            v = 1.0 - np.clip(v_top, 0.0, 1.0)
            
            gx = u * self.grid_n.get()
            gy = v * self.grid_n.get()
            
            # Calcular ángulo del robot (optimizado)
            p_dir_img = (c_robot[0] + 0.5*(c_robot[1] - c_robot[0]))
            u0, v0_top = self.pix_to_grid_uv_fast(center_px, H_inv)
            u1, v1_top = self.pix_to_grid_uv_fast(p_dir_img, H_inv)
            
            du, dv = (u1 - u0), (1.0 - v1_top) - (1.0 - v0_top)
            angle_grid = np.degrees(np.arctan2(dv, du))
            
            # Actualizar estado del robot (cache)
            self._robot_state.update({
                "detected": True,
                "position": (gx, gy),
                "heading": angle_grid,
                "timestamp": current_time,
                "px_position": (px, py)
            })
            
            self._current_robot_heading = angle_grid
            
            # Visualizar robot (menos frecuentemente)
            if self.frame_counter % self.process_every_n == 0:
                cv2.circle(frame, (px, py), 8, (0,0,255), -1)
                if self.frame_counter % (self.process_every_n * 2) == 0:
                    cv2.putText(frame, f"Robot({gx:.1f},{gy:.1f}) {angle_grid:.0f}°",
                               (px+15, py-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            
            self._last_robot_px = (px, py)
            
            # Actualizar estado en GUI (menos frecuentemente)
            if self.frame_counter % (self.process_every_n * 2) == 0:
                pos_text = f"({gx:.2f}, {gy:.2f}) @ {angle_grid:.1f}°"
                self.root.after(0, lambda: self.robot_position.set(pos_text))
            
            # Control del robot
            if self._target_g is not None:
                self.goto_controller_step(gx, gy, angle_grid)
                
        except Exception as e:
            print(f"[ERROR] Error procesando robot: {e}")
        
    def update_homography_cache_optimized(self, centers_px, current_time):
        """Actualizar cache de homografía optimizado"""
        # Cache más duradero para mejor rendimiento
        if current_time - self._homography_cache["timestamp"] < self._homography_cache["valid_duration"]:
            if self._homography_cache["H"] is not None:
                return self._homography_cache["H"], self._homography_cache["H_inv"]
        
        # Verificar si las posiciones cambiaron significativamente
        if (self._homography_cache["corner_positions"] is not None and 
            len(centers_px) == len(self._homography_cache["corner_positions"])):
            
            position_change = False
            for key in centers_px:
                if key in self._homography_cache["corner_positions"]:
                    old_pos = self._homography_cache["corner_positions"][key]
                    new_pos = centers_px[key]
                    if np.linalg.norm(new_pos - old_pos) > 3.0:  # Cambio > 3 píxeles
                        position_change = True
                        break
                        
            if not position_change:
                return self._homography_cache["H"], self._homography_cache["H_inv"]
        
        # Recalcular homografía solo si es necesario
        if len(centers_px) == 4:
            try:
                pts_dst = self.order_quad([centers_px[k] for k in centers_px.keys()])
                pts_src = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]], dtype=np.float32)
                H = cv2.getPerspectiveTransform(pts_src, pts_dst)
                H_inv = np.linalg.inv(H)
                
                self._homography_cache.update({
                    "H": H,
                    "H_inv": H_inv,
                    "timestamp": current_time,
                    "corner_positions": centers_px.copy()
                })
                
                return H, H_inv
            except Exception as e:
                print(f"[WARN] Error calculando homografía: {e}")
        
        return self._homography_cache["H"], self._homography_cache["H_inv"]
    
    def order_quad(self, pts):
        """Ordenar cuadrilátero optimizado"""
        pts = np.asarray(pts, dtype=np.float32)
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        return pts[sorted_indices]
    
    def draw_grid_homography_optimized(self, img, H):
        """Dibujar malla de homografía optimizada"""
        if H is None:
            return
            
        try:
            N = self.grid_n.get()
            step = 1.0 / N
            
            # Dibujar menos líneas para mejor rendimiento
            step_size = max(2, N // 8)  # Máximo 8 líneas por dirección
            
            # Líneas verticales
            for i in range(0, N+1, step_size):
                u = i * step
                src = np.array([[[u, 0.0]], [[u, 1.0]]], dtype=np.float32)
                dst = cv2.perspectiveTransform(src, H)
                p1, p2 = tuple(dst[0,0].astype(int)), tuple(dst[1,0].astype(int))
                cv2.line(img, p1, p2, (0,255,0), 1)
            
            # Líneas horizontales
            for i in range(0, N+1, step_size):
                v = i * step
                src = np.array([[[0.0, v]], [[1.0, v]]], dtype=np.float32)
                dst = cv2.perspectiveTransform(src, H)
                p1, p2 = tuple(dst[0,0].astype(int)), tuple(dst[1,0].astype(int))
                cv2.line(img, p1, p2, (0,255,0), 1)

            # Borde principal
            box = np.array([[[0,0]], [[1,0]], [[1,1]], [[0,1]], [[0,0]]], dtype=np.float32)
            box_t = cv2.perspectiveTransform(box, H).astype(int)
            cv2.polylines(img, [box_t.reshape(-1,2)], isClosed=True, color=(0,180,255), thickness=2)
            
        except Exception as e:
            print(f"[WARN] Error dibujando malla optimizada: {e}")
    
    def draw_target_marker_optimized(self, img):
        """Dibujar marcador de objetivo optimizado"""
        if self._target_g is None or self.CURRENT_H is None:
            return

        try:
            px = self.grid_to_pix_fast(self._target_g[0], self._target_g[1])
            if px is None:
                return

            x, y = px
            cv2.circle(img, (x, y), 12, (0, 255, 255), 3)
            cv2.line(img, (x-16, y), (x+16, y), (0, 255, 255), 2)
            cv2.line(img, (x, y-16), (x, y+16), (0, 255, 255), 2)
            
            # Texto menos frecuentemente
            if self.frame_counter % (self.process_every_n * 2) == 0:
                cv2.putText(img, f"Obj ({self._target_g[0]:.1f},{self._target_g[1]:.1f})",
                            (x+15, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            
            # Línea robot -> objetivo
            if self._last_robot_px is not None:
                cv2.line(img, self._last_robot_px, px, (0, 200, 255), 2)
        except Exception as e:
            print(f"[WARN] Error dibujando objetivo: {e}")
    
    def draw_hud_optimized(self, img):
        """Dibujar HUD de información optimizado"""
        try:
            # HUD simplificado para mejor rendimiento
            hud_lines = [
                f"Robot ID:{self.robot_id.get()} Malla:{self.grid_n.get()}x{self.grid_n.get()}",
                f"Modo: {'Orient' if self.orient_mode.get() else 'Fijo'}",
            ]
            
            y_start = 30
            for i, line in enumerate(hud_lines):
                y = y_start + i * 25
                # Sombra simplificada
                cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                # Texto principal
                cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        except Exception as e:
            print(f"[WARN] Error dibujando HUD: {e}")
    
    def display_frame_tkinter_optimized(self, frame):
        """Mostrar frame en tkinter de forma optimizada"""
        try:
            if frame is None:
                return
                
            # Obtener dimensiones del widget
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
            scale = min(scale_x, scale_y) * 0.95  # 5% de margen
            
            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)
            
            # Redimensionar solo si es necesario
            if new_width > 0 and new_height > 0 and (new_width != frame_width or new_height != frame_height):
                # Usar interpolación más rápida para display
                resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            else:
                resized_frame = frame
                
            # Convertir de BGR a RGB de forma eficiente
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            
            # Convertir a PIL Image
            pil_image = Image.fromarray(rgb_frame)
            
            # Convertir a PhotoImage para tkinter
            photo = ImageTk.PhotoImage(pil_image)
            
            # Actualizar el label en el hilo principal
            self.root.after(0, self.update_video_label, photo)
                
        except Exception as e:
            print(f"[ERROR] Error mostrando frame optimizado: {e}")
    
    def update_video_label(self, photo):
        """Actualizar el label del video en el hilo principal"""
        try:
            if self.running:  # Solo actualizar si el sistema está corriendo
                self.video_label.configure(image=photo, text="")
                self.video_label.image = photo  # Mantener referencia
        except Exception as e:
            print(f"[ERROR] Error actualizando video label: {e}")
    
    def pix_to_grid_uv_fast(self, pt_pix, H_inv):
        """Conversión rápida pixel -> UV optimizada"""
        if H_inv is None:
            return 0.0, 0.0
        try:
            x, y = float(pt_pix[0]), float(pt_pix[1])
            
            # Cálculo matricial optimizado
            h00, h01, h02 = H_inv[0, 0], H_inv[0, 1], H_inv[0, 2]
            h10, h11, h12 = H_inv[1, 0], H_inv[1, 1], H_inv[1, 2]
            h20, h21, h22 = H_inv[2, 0], H_inv[2, 1], H_inv[2, 2]
            
            w = h20*x + h21*y + h22
            if abs(w) < 1e-8:
                return 0.0, 0.0
                
            u = (h00*x + h01*y + h02) / w
            v = (h10*x + h11*y + h12) / w
            return u, v
        except:
            return 0.0, 0.0
    
    def grid_to_pix_fast(self, gx, gy):
        """Conversión rápida grid -> pixel optimizada"""
        if self.CURRENT_H is None:
            return None
        try:
            u = gx / float(self.grid_n.get())
            v_top = 1.0 - (gy / float(self.grid_n.get()))
            
            # Cálculo matricial optimizado
            H = self.CURRENT_H
            h00, h01, h02 = H[0, 0], H[0, 1], H[0, 2]
            h10, h11, h12 = H[1, 0], H[1, 1], H[1, 2]
            h20, h21, h22 = H[2, 0], H[2, 1], H[2, 2]
            
            w = h20*u + h21*v_top + h22
            if abs(w) < 1e-8:
                return None
                
            x = (h00*u + h01*v_top + h02) / w
            y = (h10*u + h11*v_top + h12) / w
            return int(x), int(y)
        except:
            return None
    
    # ==========================
    #  Métodos de control del robot optimizados
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
    
    def stop_robot(self):
        """Parar robot"""
        if self.instr_ref is None:
            return
        try:
            self.instr_ref.update({"parar": True})
            self.instr_ref.child("movimiento").update({"vx": "0", "vy": "0"})
            self.instr_ref.child("rotación").update({"w": "0"})
            print("[GOTO] Robot detenido")
        except Exception as e:
            print(f"[GOTO][ERR] Error deteniendo robot: {e}")
    
    def _send_cmd(self, vx, vy, w):
        """Enviar comando al robot optimizado"""
        if self.instr_ref is None:
            return
        try:
            # Convertir a enteros para reducir tráfico
            vx_int, vy_int, w_int = int(round(vx)), int(round(vy)), int(round(w))
            
            # Solo enviar si hay cambio significativo (reducir tráfico Firebase)
            if not hasattr(self, '_last_cmd'):
                self._last_cmd = (0, 0, 0)
                
            last_vx, last_vy, last_w = self._last_cmd
            if (abs(vx_int - last_vx) > 2 or abs(vy_int - last_vy) > 2 or abs(w_int - last_w) > 2):
                self.instr_ref.child("movimiento").update({"vx": str(vx_int), "vy": str(vy_int)})
                self.instr_ref.child("rotación").update({"w": str(w_int)})
                self._last_cmd = (vx_int, vy_int, w_int)
        except Exception as e:
            print(f"[GOTO][ERR] Error enviando comando: {e}")
    
    def _sat(self, x, lim):
        """Saturar valor optimizado"""
        return max(-lim, min(lim, x))
    
    def goto_controller_step(self, gx, gy, heading_deg):
        """Paso del controlador goto optimizado"""
        if self._target_g is None:
            return

        ex = self._target_g[0] - gx
        ey = self._target_g[1] - gy
        dist_sq = ex*ex + ey*ey  # Evitar sqrt cuando sea posible
        
        tolerance = self.pos_tolerance.get()
        if dist_sq <= tolerance*tolerance:
            self._send_cmd(0, 0, 0)
            return

        # Control proporcional lineal
        kp = self.kp_lin.get()
        vx_cmd = kp * ex
        vy_cmd = kp * ey

        # Limitar norma del vector
        norm_sq = vx_cmd*vx_cmd + vy_cmd*vy_cmd
        cmd_max = self.cmd_max.get()
        cmd_max_sq = cmd_max * cmd_max
        
        if norm_sq > cmd_max_sq:
            norm = math.sqrt(norm_sq)
            scale = cmd_max / norm
            vx_cmd *= scale
            vy_cmd *= scale

        th = math.radians(heading_deg)

        # Comando de giro optimizado
        if self.orient_mode.get():
            if norm_sq > 1e-6:  # Evitar división por cero
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

        # Enviar comando
        self._send_cmd(vx_cmd, vy_cmd, w_cmd)
    
    # ==========================
    #  Métodos de cierre
    # ==========================
    def on_closing(self):
        """Manejar cierre de aplicación"""
        if self.running:
            self.stop_system()
        
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
    """Función principal de la aplicación optimizada"""
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