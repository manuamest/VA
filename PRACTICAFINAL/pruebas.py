import cv2
import numpy as np

# Cargar la imagen
image = cv2.imread('Material_Futbol/99.jpg')

# Convertir a espacio de color HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Definir rangos de color verde
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])

# Aplicar máscara
mask = cv2.inRange(hsv, lower_green, upper_green)
cv2.imshow('Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Tratamiento de imagen para limpiar de ruido la zona del campo y tener un contorno mas limpio (Closing)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Busqueda del contorno mas grande (tiene que corresponder al campo)
blank = np.zeros(image.shape[:2], dtype='uint8')
contours, _ = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(blank, [max(contours, key=cv2.contourArea)], -1, 255, thickness= -1)

# Cruzamos la imagen original con la mascara para obtener el resultado final
resultado = cv2.bitwise_and(image, image, mask=blank)




# # Detectar bordes
# edges = cv2.Canny(mask, 50, 150)

# # Detección de líneas
# lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=5)

# # Dibujar líneas amarillas
# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

# Mostrar resultados
cv2.imshow('Resultado', resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()
