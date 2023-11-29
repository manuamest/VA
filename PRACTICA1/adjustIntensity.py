import cv2
import numpy as np

def adjustIntensity(inImage, inRange=[], outRange=[0, 1]):

    # Verificar si el rango de entrada está vacío, en ese caso calcularlo
    if not inRange:
        inRange = [np.min(inImage), np.max(inImage)]
    
    # Compresión o estiramiento  del histograma
    outImage = outRange[0] + (outRange[1] - outRange[0]) * (inImage - inRange[0]) / (inRange[1] - inRange[0])
    
    # Asegura que los valores estén en el rango [0, 1]
    outImage = np.clip(outImage, 0, 1)
    
    return outImage


def run_adjust_intensity(inImage):

    # Aplicar la función adjustIntensity con los valores deseados de inRange y outRange
    #inRange = [0, 0.9]  # Rango de entrada
    outRange = [0.1, 0.9]  # Rango de salida
    outImage = adjustIntensity(inImage, outRange)

    # Guardar la imagen de salida
    cv2.imwrite('resultados/adjustIntensity.jpg', (outImage * 255).astype(np.float32))

    # Muestra la imagen de salida y la original
    cv2.imshow('Original Intensity', inImage)
    cv2.imshow('Adjust Intensity', outImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


