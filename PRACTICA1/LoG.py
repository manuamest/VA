import cv2
import numpy as np
from normalizacionminmax import normalizar_imagen
from filterImage import filterImage
from gaussianFilter import gaussKernel1D, gaussianFilter

def LoG(inImage, sigma):

    laplace = np.array([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]], dtype=np.float32)
    
    gaussianImage = gaussianFilter(inImage, sigma)

    outImage = filterImage(gaussianImage, laplace)

    return outImage

def run_LoG(inImage):
    
    #image = normalizar_imagen(inImage)

    sigma = 1.5

    # Aplicar el filtro a la imagen
    output_image = LoG(inImage, sigma)

    # Guardar imagen    
    cv2.imwrite('resultados/LoG.jpg', (output_image * 255).astype(np.float32))

    # Mostrar la imagen original y la filtrada
    cv2.imshow("Original Image", inImage)
    cv2.imshow("LoG Image", output_image)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()