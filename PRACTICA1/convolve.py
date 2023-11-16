import numpy as np
import cv2

def gradientImage(inImage, operator):
    if operator == 'Roberts':
        kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
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

    gx = convolve(inImage, kernel_x)
    gy = convolve(inImage, kernel_y)

    return gx, gy

def convolve(image, kernel):
    # Obtener las dimensiones del kernel
    kH, kW = kernel.shape[:2]

    # Añadir padding a la imagen de entrada
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    # Preparar la imagen de salida
    output = np.zeros_like(image, dtype="float32")

    # Aplicar la convolución
    for y in np.arange(pad, image.shape[0] - pad):
        for x in np.arange(pad, image.shape[1] - pad):
            # Extraer el vecindario de la imagen de entrada
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            # Realizar la convolución
            k = (roi * kernel).sum()

            # Guardar el resultado en la imagen de salida
            output[y - pad, x - pad] = k

    # Normalizar la imagen de salida para que esté en el rango [0, 255]
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    return output
