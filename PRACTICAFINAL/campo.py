import cv2 as cv
import numpy as np
from skimage import color, measure

offset = 2

imagen = cv.imread('Material_Futbol/10.jpg')

# Imagen de RGB a HSV para calcular histograma de verdes en el canal H (Util para trabajar con colores)
hsvImage = cv.cvtColor(imagen,cv.COLOR_BGR2HSV)

# Calculo de maximos relativos de color verde predominante
hist1 = cv.calcHist([hsvImage],[0],None,[256],[0,256])
hist_green = hist1[:120]

# Valor del maximo absoluto
h_max = np.argmax(hist_green)

# Filtro para considerar tonos de verde
h_rel_filter = hist_green[h_max] * 0.03

h_maximos = []
h_min_filter = 0
h_max_filter = 150
if h_max > 30:
    h_min_filter = h_max - 30
if h_max < 120:
    h_max_filter = h_max + 30
for x in range(h_min_filter, h_max_filter):
    if (hist_green[x] >= h_rel_filter):
        h_maximos.append(x)

# Se aplica al menor y mayor maximo relativo porque en caso de sombras en el campo hay que coger un espectro mayor
threshold_min = min(h_maximos) - offset
threshold_max = max(h_maximos) + offset

#Aplicacion de la mascara con los verdes mas repetidos
lower_region= np.array([threshold_min,0,0])
upper_region = np.array([threshold_max,255,255])
mask = cv.inRange(hsvImage, lower_region, upper_region)
############################################################################################# HASTA AQUI LO HACE MUY BIEN

# Tratamiento de imagen para limpiar de ruido la zona del campo y tener un contorno mas limpio (Closing)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (21, 21))
closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

# Busqueda del contorno mas grande (tiene que corresponder al campo)
blank = np.zeros(imagen.shape[:2], dtype='uint8')
contours, hierarchies = cv.findContours(closing, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
cv.drawContours(blank, [max(contours, key=cv.contourArea)], -1, 255, thickness=-1)

# Cruzamos la imagen original con la mascara para obtener el resultado final
resultado = cv.bitwise_and(imagen, imagen, mask=blank)

image3 = resultado

cv.imshow('Original', imagen)
cv.imshow("field_area_mask", image3)
cv.waitKey(0)
cv.destroyAllWindows()
