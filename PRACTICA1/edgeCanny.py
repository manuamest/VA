import cv2
import numpy as np
from gaussianFilter import gaussianFilter
from gradientImage import gradientImage

def edgeCanny(inImage, sigma, tlow, thigh):
    # Filtro Gaussiano para suavizar la imagen
    smoothed_image = gaussianFilter(inImage, sigma)

    # Calcular gradientes en las direcciones x e y con el operador Sobel
    [gradient_x, gradient_y] = gradientImage(smoothed_image, 'Sobel')

    # Calcular la magnitud del gradiente y la dirección
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = (np.arctan2(gradient_y, gradient_x) * (180 / np.pi)) 

    # Supresión no máxima
    suppressed_image = np.zeros_like(gradient_magnitude)
    for i in range(1, gradient_magnitude.shape[0] - 1):
        for j in range(1, gradient_magnitude.shape[1] - 1):
            gradient_direction[gradient_direction < 0] += 180
            direction = gradient_direction[i, j]
            if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
                neighbors = [gradient_magnitude[i, j - 1], gradient_magnitude[i, j + 1]]
            elif (22.5 <= direction < 67.5):
                neighbors = [gradient_magnitude[i - 1, j - 1], gradient_magnitude[i + 1, j + 1]]
            elif (67.5 <= direction < 112.5):
                neighbors = [gradient_magnitude[i - 1, j], gradient_magnitude[i + 1, j]]
            elif (112.5 <= direction < 157.5):
                neighbors = [gradient_magnitude[i - 1, j + 1], gradient_magnitude[i + 1, j - 1]]

            # Comparar magnitud actual con vecinos en la dirección del gradiente
            if gradient_magnitude[i, j] >= max(neighbors):
                suppressed_image[i, j] = gradient_magnitude[i, j]

    # Umbral de histéresis
    edges = np.zeros_like(suppressed_image)
    strong_edges = (suppressed_image >= thigh)
    weak_edges = (suppressed_image >= tlow) & (suppressed_image <= thigh)

    # Mantener bordes fuertes
    edges[strong_edges] = 1

    # Etapa de seguimiento de bordes débiles conectados a bordes fuertes
    for i in range(1, suppressed_image.shape[0] - 1):
        for j in range(1, suppressed_image.shape[1] - 1):
            gradient_direction[gradient_direction < 0] += 180
            if weak_edges[i, j]:
                # Obtener vecinos en la dirección del gradiente
                direction = gradient_direction[i, j]
                if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
                    neighbors = [(i, j - 1), (i, j + 1)]
                elif (22.5 <= direction < 67.5):
                    neighbors = [(i - 1, j - 1), (i + 1, j + 1)]
                elif (67.5 <= direction < 112.5):
                    neighbors = [(i - 1, j), (i + 1, j)]
                elif (112.5 <= direction < 157.5):
                    neighbors = [(i - 1, j + 1), (i + 1, j - 1)]

    # Verificar vecinos para conexión a bordes fuertes
    if np.any(strong_edges[x, y] for x, y in neighbors):
        edges[i, j] = 1
    
    return edges

def run_edgeCanny(inImage):
    # Parámetros de Canny
    sigma = 0.2
    tlow = 0.05
    thigh = 0.2

    # Aplicar el detector de bordes de Canny
    outImage = edgeCanny(inImage, sigma, tlow, thigh)

    # Guardar imagen resultado
    cv2.imwrite('resultados/canny.jpg', (outImage * 255).astype(np.float32))

    # Mostrar la imagen original y el resultado
    cv2.imshow('Original Image', inImage)

    cv2.imshow('Canny Edge Detection', outImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
