import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import os

# Ruta absoluta del directorio donde está el script
path_1 = os.path.dirname(os.path.abspath(__file__))
path_2 = os.path.join(path_1, "Codigos_ArUco")

# Seleccionar el diccionario de marcadores de ArUco
# DICT_4x4_50, DICT_5x5_100, DICT_6x6_250, DICT_7x7_1000, DICT_ARUCO_ORIGINAL -> 1024
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Genera el marcador con ID # y tamaño ###x### pixeles
marker_id = [20, 21, 23, 24]
marker_size = 250

for x in marker_id:
    marker_image = aruco.generateImageMarker(aruco_dict, x, marker_size)

    # Muestra el marcador con matplotlib
    plt.imshow(marker_image, cmap='gray')
    plt.axis('off')
    plt.title(f'ArUco ID {x}')
    plt.show()

    # Guardar el marcador como imagen PNG
    aruco_filename = os.path.join(path_2, f"Codigo_Aruco_{x}.jpg")
    cv2.imwrite(aruco_filename, marker_image)
    #cv2.imwirte(f'aruco_{marker_id}.png', marker_image)