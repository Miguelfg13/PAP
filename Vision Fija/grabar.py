import cv2
import os

dir = os.path.dirname(os.path.abspath(__file__))
dir_video = os.path.join(dir, "videoSalida.avi")
captura = cv2.VideoCapture(1)

salida = cv2.VideoWriter(dir_video,cv2.VideoWriter_fourcc(*'XVID'),30.0,(1280,720))
while (captura.isOpened()):
  ret, imagen = captura.read()
  if ret == True:
    cv2.imshow('video', imagen)
    salida.write(imagen)
    if cv2.waitKey(1) & 0xFF == ord('s'):
      break
  else: break
captura.release()
salida.release()
cv2.destroyAllWindows()