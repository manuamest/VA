import cv2
import numpy as np
from filterImage import filterImage

def gradientImage(inImage, operator):

    # Seleccion de operador
    if operator == 'Roberts':
        kernel_x = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    elif operator == 'CentralDiff':
        kernel_x = np.array([[-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1], [0], [1]], dtype=np.float32)
    elif operator == 'Prewitt':
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    elif operator == 'Sobel':
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    else:
        return "Error: operador no reconocido"
    
    # Calculo gx y gy
    gx = filterImage(inImage, kernel_x)
    gy = filterImage(inImage, kernel_y)

    return gx, gy

def run_gradientImage(inImage, op):

    # Aplicar el operador
    gx, gy = gradientImage(inImage, op)

    # Guardar los resultados
    cv2.imwrite('resultados/gx.png', (gx * 255).astype(np.float32))
    cv2.imwrite('resultados/gy.png', (gy * 255).astype(np.float32))

    # Mostrar las imágenes resultantes
    cv2.imshow("Original Image", inImage)
    cv2.imshow("Gradient X", gx)
    cv2.imshow("Gradient Y", gy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()