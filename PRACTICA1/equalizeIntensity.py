import cv2
import numpy as np

def equalizeIntensity(inImage, nBins=256):

    # Asegura que la imagen está en el rango [0, 1]
    inImage = np.clip(inImage, 0, 1)

    # Crear un histograma acumulativo
    hist, _ = np.histogram(inImage * (nBins - 1), bins=nBins, range=(0, nBins - 1)) # Calculo del histograma
    histacum = hist.cumsum()                                                        # Histograma acumulativo
    histacum = histacum / histacum[-1]                                              # Normaliza entre 0 1

    # Ecualizar la imagen usando el histograma acumulativo
    outImage = histacum[np.round(inImage * (nBins - 1)).astype(int)]
    
    # Asegurarse de que la imagen de salida está en el rango [0, 1]
    outImage = np.clip(outImage, 0, 1)

    return outImage

def run_equalize_intensity(inImage):

    # Aplica la ecualización de intensidad
    output_image = equalizeIntensity(inImage, 256)

    cv2.imwrite('resultados/equalizeIntensity.jpg', (output_image * 255).astype(np.float32))

    # Muestra la imagen de entrada y la imagen de salida
    cv2.imshow('Imagen de Entrada', inImage)
    cv2.imshow('Imagen de Salida', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()