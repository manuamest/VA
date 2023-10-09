import cv2
import numpy as np

def adjustIntensity(inImage, inRange=[], outRange=[0, 1]):

    # Verificar si el rango de entrada está vacío, en ese caso calcularlo
    if not inRange:
        inRange = [np.min(inImage), np.max(inImage)]
    
    # Aplicar la compresión o estiramiento lineal del histograma
    outImage = outRange[0] + (outRange[1] - outRange[0]) * (inImage - inRange[0]) / (inRange[1] - inRange[0])
    
    # Asegurarse de que los valores estén en el rango [0, 1]
    outImage = np.clip(outImage, 0, 1)
    
    return outImage


# Ejemplo de uso:
if __name__ == "__main__":
    # Cargar la imagen de entrada (asegúrate de que sea de punto flotante)
    inImage = cv2.imread('/home/manuamest/Documentos/UNI/CUARTO/PRIMER CUATRI/VA/PRACTICA1/imgp1/imagen_normalizada.png', cv2.IMREAD_GRAYSCALE)
    imagen_normalizada = inImage.astype(np.float32) / 255.0

    # Aplicar la función adjustIntensity con los valores deseados de inRange y outRange
    inRange = [0.3, 0.7]  # Por ejemplo, establecer el rango de entrada
    outRange = [0.1, 0.9]  # Establecer el nuevo rango de salida
    outImage = adjustIntensity(imagen_normalizada, inRange, outRange)

    # Guardar la imagen de salida
    cv2.imwrite('imgp1/adjustIntensity.jpg', (outImage * 255).astype(np.uint8))
    cv2.imshow('Adjust Intensity', outImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


