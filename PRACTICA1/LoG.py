import cv2
import numpy as np
from filterImage import filterImage
from gaussianFilter import gaussKernel1D, gaussianFilter

def LoG(inImage, sigma):

    # Laplace 3x3
    laplace = np.array([[ 0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)

    # Laplace 5x5
    laplace = np.array([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]])

    # Filtro de gauss
    gaussianImage = gaussianFilter(inImage, sigma)

    # Filter image usando laplace como kernel
    laplaceImage = filterImage(gaussianImage, laplace)

    # Calcula un umbral basado en la media de los valores absolutos
    t = np.abs(laplaceImage).mean() * 2.5

    # Crea una imagen de salida binaria
    outImage = np.zeros(laplaceImage.shape, dtype=np.uint8)

    # Encuentra puntos de interés
    for i in range(1, outImage.shape[0] - 1):
        for j in range(1, outImage.shape[1] - 1):

            neighbourhood = laplaceImage[i-1:i+2, j-1:j+2]

            if laplaceImage[i, j] < -t and np.any(neighbourhood) > t:
                outImage[i, j] = 255

    return outImage

def run_LoG(inImage):
    
    sigma = 1

    # Aplicar el filtro a la imagen
    output_image = LoG(inImage, sigma)

    # Guardar imagen    
    cv2.imwrite('resultados/LoG.jpg', (output_image * 255).astype(np.float32))

    # Mostrar la imagen original i la filtrada
    cv2.imshow("Original Image", inImage)
    cv2.imshow("LoG Image", output_image)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()