import cv2
import numpy as np

def filterImage(inImage, kernel):
    # Obtener las dimensiones de la imagen de entrada y del kernel
    rows, cols = inImage.shape
    kRows, kCols = kernel.shape if len(kernel.shape) != 1 else (1, len(kernel))
    
    # Calcular el desplazamiento necesario para centrar el kernel
    dRow = kRows // 2
    dCol = kCols // 2

    outImage = inImage.copy()

    # Convolución
    for i in range(dRow, rows - dRow):
        for j in range(dCol , cols - dCol):
            # Extraer la región de interés de la imagen de entrada
            if len(kernel.shape) == 1:
                rdi = inImage[i, j - dCol:j + dCol + 1]
            else:
                rdi = inImage[i - dRow:i + dRow + 1, j - dCol:j + dCol + 1]

            # Aplicar la convolución entre el kernel y la región de interés
            conv = np.sum(rdi * kernel)

            # Asignar el resultado a la posición correspondiente en la imagen de salida
            outImage[i, j] = conv

    return outImage

def run_filterImage(inImage):

    # Definir un kernel
    #kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    kernel = np.array([1, 1, 1])

    # Aplicar el filtro a la imagen
    output_image = filterImage(inImage, kernel.T)

    # Guardar la imagen resultado
    cv2.imwrite('resultados/filterImage2.jpg', (output_image * 255).astype(np.float32))

    # Mostrar la imagen original y la filtrada
    cv2.imshow("Original Image", inImage)
    cv2.imshow("Filtered Image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
