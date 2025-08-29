import numpy as np
import matplotlib.pyplot as plt

""" nx, ny = 7, 6
grid = np.mgrid[0:nx, 0:ny]
print("Shape original:", grid.shape)

grid_T = grid.T
print("Shape transpuesta:", grid_T.shape)

points = grid_T.reshape(-1, 2)
print("Lista de puntos 2D:")
print(points) """

# Dimensiones del patrón (por ejemplo, tablero de ajedrez de 7x6)
nx, ny = 7, 6

# Sin transpuesta
grid = np.mgrid[0:nx, 0:ny]  # shape (2, 7, 6)
points_bad = grid.reshape(-1, 2)  # Mal emparejado

# Con transpuesta
points_good = grid.T.reshape(-1, 2)  # Bien emparejado

# Graficar
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Sin .T (mal ordenado)
axs[0].scatter(points_bad[:, 0], points_bad[:, 1], c='red')
axs[0].set_title("Sin .T (coordenadas mal emparejadas)")
axs[0].set_aspect('equal')
axs[0].invert_yaxis()
axs[0].grid(True)

# Con .T (bien ordenado)
axs[1].scatter(points_good[:, 0], points_good[:, 1], c='green')
axs[1].set_title("Con .T (coordenadas correctamente emparejadas)")
axs[1].set_aspect('equal')
axs[1].invert_yaxis()
axs[1].grid(True)

plt.suptitle("Comparación: reshape() con y sin transpuesta", fontsize=14)
plt.tight_layout()
plt.show()