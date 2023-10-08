import cv2
import numpy as np

def equalizeIntensity(inImage, nBins=256):
    # Asegurarse de que la imagen está en el rango [0, 1]
    inImage = np.clip(inImage, 0, 1)

    # Crear un histograma acumulativo
    hist, _ = np.histogram(inImage * (nBins - 1), bins=nBins, range=(0, nBins - 1))
    cdf = hist.cumsum()
    cdf = cdf / cdf[-1]

    # Ecualizar la imagen usando el histograma acumulativo
    outImage = cdf[np.round(inImage * (nBins - 1)).astype(int)]
    
    # Asegurarse de que la imagen de salida está en el rango [0, 1]
    outImage = np.clip(outImage, 0, 1)

    return outImage

# Carga tu imagen de escala de grises en [0, 1]
input_image = cv2.imread('imgp1/imagen_normalizada.png', cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

# Aplica la ecualización de intensidad
output_image = equalizeIntensity(input_image)

cv2.imwrite('imgp1/equalizeIntensity.jpg', (output_image * 255).astype(np.uint8))

# Muestra la imagen de entrada y la imagen de salida
cv2.imshow('Imagen de Entrada', input_image)
cv2.imshow('Imagen de Salida', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
