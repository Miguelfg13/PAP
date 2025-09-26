import cv2
import cv2.aruco as aruco
import pandas as pd
import numpy as np
import json
import os

ARCHIVO_PARAMETROS_CAMARA = "parametros_camara.json"
ARCHIVO_VALORES_CSV ="valores.csv"

def obtener_Direccion_Carpeta_Absoluta():
    return os.path.dirname(os.path.abspath(__file__))

def juntar_Direccion_Archivo(direccion_carpeta, nombre_archivo):
    return os.path.join(direccion_carpeta, nombre_archivo)

def guardar_Parametros_Camara(
        resolucion_camara, matriz_camara, coeficientes_distorsion, fecha_calibracion
        ):
    
    resolucion = resolucion_camara.split('x')
    datos = {
        "fecha_calibracion" : str(fecha_calibracion),
        "resolucion_width" : int(resolucion[0]),
        "resolucion_height" : int(resolucion[1]),
        "matriz_camara" : matriz_camara.tolist(),
        "coef_dist" : coeficientes_distorsion.tolist()
    }

    direccion = juntar_Direccion_Archivo(obtener_Direccion_Carpeta_Absoluta(), ARCHIVO_PARAMETROS_CAMARA)
    
    with open(direccion, "w") as f:
        json.dump(datos, f, indent=4) 

def gurdar_Parametros_Extrinsecos_Camara(matriz_rotacion, vector_traslancion):
    datos = {
        "matriz_rotacion" : matriz_rotacion.tolist(),
        "vector_traslacion" : vector_traslancion.tolist()
    }

    parametros_camara = obtener_Parametros_Camara()
    parametros_camara.update(datos)

    direccion = juntar_Direccion_Archivo(obtener_Direccion_Carpeta_Absoluta(), ARCHIVO_PARAMETROS_CAMARA)
    
    with open(direccion, "w") as f:
        json.dump(parametros_camara, f, indent = 4)

def obtener_Parametros_Camara():
    direccion = juntar_Direccion_Archivo(obtener_Direccion_Carpeta_Absoluta(), ARCHIVO_PARAMETROS_CAMARA)

    if not os.path.exists(direccion):
        raise FileNotFoundError(f"No se encontró el archivo de parámetros: {direccion}")

    with open(direccion, "r") as f:
        parametros_camara = json.load(f)

    return parametros_camara

def guardar_Datos_CSV(datos):
    if not datos:
        raise ValueError(f"Los datos proporcionados para guardar en CSV están vacíos.")

    direccion = juntar_Direccion_Archivo(obtener_Direccion_Carpeta_Absoluta(), ARCHIVO_VALORES_CSV)
    df = pd.DataFrame(datos)
    df.to_csv(direccion, index = False)

def sacar_Datos_CSV():
    direccion = juntar_Direccion_Archivo(obtener_Direccion_Carpeta_Absoluta(), ARCHIVO_VALORES_CSV)

    if not os.path.exists(direccion):
        raise FileNotFoundError(f"No se encontró el archivo CSV: {direccion}")
    
    return pd.read_csv(direccion)