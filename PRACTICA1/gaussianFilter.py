import numpy as np
import cv2
from filterImage import filterImage

def gaussKernel1D(sigma):
    # Calcular N a partir de σ
    N = int(2 * np.ceil(3 * sigma) + 1)

    # Calcular el centro del kernel
    center = N // 2

    # Inicializar el kernel como un vector de ceros
    kernel = np.zeros(N)

    # Calcular el valor del kernel Gaussiano
    for i in range(N):
        x = i - center
        kernel[i] = np.exp(-0.5 * (x / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

    # Normalizar el kernel para que la suma sea igual a 1
    kernel /= np.sum(kernel)

    return kernel

def gaussianFilter(inImage, sigma):
    # Calcular los kernels unidimensionales
    kernel1D = gaussKernel1D(sigma)

    # Aplicar la convolución horizontal (1D)
    filtered_horizontal = filterImage(inImage, kernel1D)

    # Aplicar la convolución vertical (1D) al resultado de la convolución horizontal
    filtered_image = filterImage(np.transpose(filtered_horizontal), np.transpose(kernel1D))

    return np.transpose(filtered_image)

def run_gaussianFilter(inImage):
    # Parámetro σ
    sigma = 3

    # Aplica el filtro gaussiano
    output_image = gaussianFilter(inImage, sigma)

    cv2.imwrite('resultados/gaussianFilter.jpg', (output_image * 255).astype(np.float32))

    # Muestra la imagen de entrada y la imagen de salida
    cv2.imshow('Original image', inImage)
    cv2.imshow('Gaussian Image', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()