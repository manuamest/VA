import cv2
import numpy as np

def cornerSusan(inImage, r, t):
    # Convierte la imagen de entrada a escala de grises si no lo está
    if len(inImage.shape) == 3:
        inImage = cv2.cvtColor(inImage, cv2.COLOR_BGR2GRAY)

    # Obtén las dimensiones de la imagen
    rows, cols = inImage.shape

    # Inicializa las matrices de salida
    outCorners = np.zeros_like(inImage, dtype=np.float32)
    susanArea = np.zeros_like(inImage, dtype=np.float32)

    # Definir la máscara circular
    y, x = np.ogrid[-r:r+1, -r:r+1]
    circular_mask = x**2 + y**2 <= r**2

    # Itera sobre cada píxel de la imagen
    for i in range(r, rows - r):
        for j in range(r, cols - r):
            # Extrae la región circular alrededor del píxel (i, j) con radio r
            region = inImage[i - r:i + r + 1, j - r:j + r + 1]

            # Calcula la diferencia de intensidad entre el píxel central y la región
            diff = np.abs(region - inImage[i, j])

            # Calcula el área SUSAN
            susan = circular_mask & (diff <= t)
            n = np.sum(susan)

            # Calcular el umbral geometrico
            geom_threshold = 3 / 4 * np.sum(circular_mask)

            # Compara el área SUSAN con un umbral y marca el píxel como esquina si se cumple
            if n < geom_threshold:  # Criterio geométrico de SUSAN
                outCorners[i, j] = 255
                susanArea[i, j] = geom_threshold - n

            # Almacena el área USAN en la matriz de salida correspondiente

    return outCorners, susanArea

# Ejemplo de uso
# Lee la imagen de entrada
image = cv2.imread('imgp1/CUADRADO.jpg', cv2.IMREAD_GRAYSCALE)

# Especifica el radio y umbral  
radio_mascara = 10
umbral_diferencia = 0.1

# Aplica el detector de esquinas SUSAN
esquinas, area_usan = cornerSusan(image, radio_mascara, umbral_diferencia)

# Muestra las imágenes de salida
# cv2.imshow('Original', image)
cv2.imshow('Esquinas', esquinas)
cv2.imshow('Area USAN', area_usan)
cv2.waitKey(0)
cv2.destroyAllWindows()
