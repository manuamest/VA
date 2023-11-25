import cv2
import numpy as np

def medianFilter(inImage, filterSize):
    # Dimensiones de la imagen de entrada
    rows, cols = inImage.shape

    # Centro
    center = filterSize // 2 

    # Crear una imagen de salida inicializada a ceros
    outImage = np.zeros_like(inImage)

    # Iterar sobre la imagen
    for i in range(rows):
        for j in range(cols):

            # Definir los límites de la ventana
            row_min = max(0, i - center)
            row_max = min(rows, i + center + 1)
            col_min = max(0, j - center)
            col_max = min(cols, j + center + 1)

            # Extraer la región de interés (ventana)
            rdi = inImage[row_min:row_max, col_min:col_max]

            # Calcular la mediana de la ventana y asignarla al píxel de salida
            outImage[i, j] = np.median(rdi)

    return outImage

def run_medianFilter(inImage):

    # Tamaño del filtro de mediana (por ejemplo, 3x3)
    filterSize = 7

    # Aplicar el filtro de medianas
    output_image = medianFilter(inImage, filterSize)

    cv2.imwrite('resultados/medianFilter.jpg', (output_image * 255).astype(np.float32))

    # Mostrar la imagen original y la filtrada
    cv2.imshow("Original Image", inImage)
    cv2.imshow("Median Filtered Image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
