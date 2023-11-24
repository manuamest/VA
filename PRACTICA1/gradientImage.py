import cv2
import numpy as np

def filterImage(inImage, kernel):
    # Obtener las dimensiones de la imagen de entrada y el kernel
    rows, cols = inImage.shape
    kRows, kCols = kernel.shape

    # Calcular el desplazamiento necesario para centrar el kernel
    dRow = kRows // 2
    dCol = kCols // 2

    # Crear una imagen de salida inicializada a ceros
    outImage = np.zeros_like(inImage, dtype=np.float32)

    # Convolución
    for i in range(dRow, rows - dRow):
        for j in range(dCol, cols - dCol):
            # Extraer la región de interés de la imagen de entrada
            roi = inImage[i - dRow:i + dRow + 1, j - dCol:j + dCol + 1]

            # Aplicar la convolución entre el kernel y la región de interés
            conv_result = np.sum(roi * kernel)

            # Asignar el resultado a la posición correspondiente en la imagen de salida
            outImage[i, j] = conv_result

    return outImage


def gradientImage(inImage, operator):

    # Seleccion de operador
    if operator == 'Roberts':
        kernel_x = np.array([[-1, 0], [0, 1]], dtype=np.float32)
        kernel_y = np.array([[0, -1], [1, 0]], dtype=np.float32)
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

if __name__ == "__main__":
    # Cargar una imagen en escala de grises (asegúrate de que está en [0, 255])
    image = cv2.imread("imgp1/CIRCULO.jpg", cv2.IMREAD_GRAYSCALE) / 255.0

    # Aplicar el operador 'Sobel' como ejemplo
    gx, gy = gradientImage(image, 'CentralDiff')

    # Mostrar las imágenes resultantes
    cv2.imshow("Original Image", image)
    cv2.imshow("Gradient X", gx)
    cv2.imshow("Gradient Y", gy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()