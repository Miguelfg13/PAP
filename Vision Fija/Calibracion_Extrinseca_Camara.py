import cv2
import numpy as np
import os

# Ruta absoluta del directorio donde está el script
script_dir = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(script_dir, "Fotos_Calibracion")
image_path = os.path.join(folder, 'foto_ajedrez_1.jpg')

if os.path.exists(image_path):
    print("Folder encontrado: ", image_path)


# --- Parámetros conocidos del tablero ---
nx = 7   # número de esquinas internas en X (ej: para 8x8 cuadros → 7x7 esquinas)
ny = 7   # número de esquinas internas en Y
square_size = 26  # tamaño de cada cuadro en mm (ajústalo al tuyo)

# --- Parámetros de la calibración intriseca ---
K = np.array([[700.27703336,  0, 344.1757582],
             [0, 704.72638774, 226.15187565],
             [0, 0, 1]])
dist = np.array([[-0.44453011,  0.20908467,  0.00520405, -0.00129481,  0.9669697]])

# --- Preparar coordenadas 3D de las esquinas del tablero ---
objp = np.zeros((nx*ny, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
objp *= square_size  # escala en milímetros

# --- Cargar una imagen del tablero ---
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Buscar esquinas ---
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

if ret:
    # Refinar esquinas
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # --- Resolver PnP para obtener R y t ---
    ret, rvec, tvec = cv2.solvePnP(objp, corners_sub, K, dist)

    # rvec = vector de rotación (Rodrigues)
    # tvec = vector de traslación

    # Convertir rvec a matriz de rotación
    R, _ = cv2.Rodrigues(rvec)

    print("Matriz de rotación (R):\n", R)
    print("Vector de traslación (t):\n", tvec)

    # --- Dibujar el eje 3D en la imagen ---
    axis = np.float32([[50,0,0], [0,50,0], [0,0,-50]])  # ejes X(rojo), Y(verde), Z(azul)
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)

    corner = tuple(corners_sub[0].ravel().astype(int))
    img = cv2.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (0,0,255), 3) # X
    img = cv2.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 3) # Y
    img = cv2.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (255,0,0), 3) # Z

    cv2.imshow("Extrinseca", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No se encontraron esquinas en la imagen.")
